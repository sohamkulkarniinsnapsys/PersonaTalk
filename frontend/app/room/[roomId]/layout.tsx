export default function RoomLayout({ children }: { children: React.ReactNode }) {
  // Intentionally minimal. Do not introduce exit/unmount delays here; the room page
  // owns WebRTC cleanup timing.
  return <>{children}</>;
}
