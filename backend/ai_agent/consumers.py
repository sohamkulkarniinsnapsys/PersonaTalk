import json
import logging
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer

logger = logging.getLogger(__name__)

class CallConsumer(AsyncWebsocketConsumer):
    # Class-level storage for managers per room to prevent duplicates
    _managers = {}
    # Track active connections per room
    _room_connection_counts = {}
    # Track if offer has been processed for each room (prevents duplicate processing)
    _room_offers_processed = {}
    # Lock to serialize manager creation (prevents race condition)
    _manager_creation_lock = asyncio.Lock()
    # Lock for connection counting
    _connection_count_lock = asyncio.Lock()
    # Lock for offer tracking
    _offer_tracking_lock = asyncio.Lock()
    
    async def connect(self):
        self.room_id = self.scope['url_route']['kwargs']['room_id']
        self.room_group_name = f'room_{self.room_id}'

        # Track this connection
        async with self._connection_count_lock:
            if self.room_id not in self._room_connection_counts:
                self._room_connection_counts[self.room_id] = 0
            self._room_connection_counts[self.room_id] += 1
            connection_num = self._room_connection_counts[self.room_id]
        
        logger.info(f"WebSocket connected: {self.room_id} (connection #{connection_num})")

        # Join room group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )

        await self.accept()

    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )
        
        # Decrement connection count and check if this is the last one
        async with self._connection_count_lock:
            if self.room_id in self._room_connection_counts:
                self._room_connection_counts[self.room_id] -= 1
                remaining = self._room_connection_counts[self.room_id]
            else:
                remaining = 0
        
        logger.info(f"WebSocket disconnected: {self.room_id} ({remaining} connections remaining)")
        
        # ONLY close manager if this was the last connection for this room
        if remaining <= 0:
            if self.room_id in self._managers:
                manager = self._managers[self.room_id]
                try:
                    await manager.close()
                    logger.info(f"Manager closed for room {self.room_id} (last connection)")
                except Exception as e:
                    logger.error(f"Error closing manager: {e}", exc_info=True)
                finally:
                    del self._managers[self.room_id]
            
            # Clean up connection counter
            if self.room_id in self._room_connection_counts:
                del self._room_connection_counts[self.room_id]
            
            # Clean up offer tracking
            if self.room_id in self._room_offers_processed:
                del self._room_offers_processed[self.room_id]
        else:
            logger.info(f"Keeping manager alive: {remaining} WebSocket connections still active for room {self.room_id}")

    # Receive message from WebSocket
    async def receive(self, text_data):
        try:
            data = json.loads(text_data)
            msg_type = data.get('type')
            
            logger.info(f"Received message type: {msg_type} in room {self.room_id}")

            if msg_type == 'offer':
                # Import here to avoid early load issues
                from .webrtc import WebRTCManager
                
                # Track if we've already processed an offer for this room
                async with self._offer_tracking_lock:
                    if self.room_id in self._room_offers_processed:
                        logger.info(f"Duplicate offer received for room {self.room_id} (from another WebSocket connection); skipping")
                        # Still need to answer to acknowledge the offer
                        # But we'll reuse the existing manager's answer
                        if self.room_id in self._managers:
                            manager = self._managers[self.room_id]
                            answer = await manager.handle_offer(data['sdp'], data['type'])
                            await self.send(text_data=json.dumps(answer))
                        return
                    else:
                        self._room_offers_processed[self.room_id] = True
                
                # CRITICAL: Use lock to serialize manager creation (prevents race condition)
                # When two WebSocket connections arrive simultaneously, only one should create the manager
                async with self._manager_creation_lock:
                    # Check AGAIN inside the lock - another connection may have created it while we waited
                    if self.room_id not in self._managers:
                        logger.info(f"Creating new WebRTCManager for room {self.room_id}")
                        # Provide a broadcast callback so manager/controllers can push UI events (transcripts)
                        async def broadcast(message: dict):
                            # Send to all sockets in this room
                            await self.channel_layer.group_send(
                                self.room_group_name,
                                {
                                    'type': 'ai_message',
                                    'message': message,
                                    'sender_channel_name': self.channel_name
                                }
                            )

                        self._managers[self.room_id] = WebRTCManager(self.room_id, self.send_json, broadcast)
                    else:
                        logger.info(f"Manager already exists for room {self.room_id}; reusing")
                
                self.webrtc_manager = self._managers[self.room_id]
                answer = await self.webrtc_manager.handle_offer(data['sdp'], data['type'])
                await self.send(text_data=json.dumps(answer))
            
            elif msg_type == 'ice-candidate':
                if self.room_id in self._managers:
                    await self._managers[self.room_id].handle_candidate(data)
                
        except json.JSONDecodeError:
            logger.error("Invalid JSON received")

    async def send_json(self, data):
        await self.send(text_data=json.dumps(data))


    # Receive message from room group
    async def signaling_message(self, event):
        message = event['message']
        sender_channel_name = event['sender_channel_name']

        # Send message to WebSocket, but not to the sender
        if self.channel_name != sender_channel_name:
            await self.send(text_data=json.dumps(message))

    async def ai_message(self, event):
        """
        Handler for messages sent by the AI agent to the client (captions, status, etc.)
        """
        message = event['message']
        await self.send(text_data=json.dumps(message))
