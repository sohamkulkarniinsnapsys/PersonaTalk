from __future__ import annotations

# Isolated deterministic question bank for interviewer persona
# Each item has canonical answer and key concepts for evaluation

QUESTION_BANK = {
    "general": {
        "basic": [
            {
                "q": "In simple terms, what is a REST API and how is it different from RPC?",
                "answer": (
                    "A REST API exposes resources over HTTP using standard verbs (GET, POST, PUT, DELETE). "
                    "Clients operate on resource representations via stateless requests, where URLs identify resources. "
                    "RPC focuses on calling functions or procedures, modeling operations as method calls rather than resources."
                ),
                "concepts": ["http verbs", "resources", "stateless", "urls", "rpc is function calls"],
                "hint": "Think about resources and standard HTTP verbs versus calling functions."
            },
            {
                "q": "What do HTTP 200 and 404 status codes mean?",
                "answer": "200 means a request succeeded. 404 means the requested resource was not found.",
                "concepts": ["200 ok", "success", "404 not found", "resource missing"],
                "hint": "One signals success; the other indicates the resource isn't there."
            },
            {
                "q": "What is the time complexity of binary search and why?",
                "answer": (
                    "O(log n) because each step halves the remaining search interval in a sorted array."
                ),
                "concepts": ["log n", "halve", "sorted"],
                "hint": "Consider how many times you can halve the search space."
            },
        ],
        "moderate": [
            {
                "q": "How would you design a rate limiter for an API? Mention one algorithm.",
                "answer": (
                    "Use token bucket or leaky bucket with a shared store (e.g., Redis) to track tokens per identity. "
                    "Requests consume tokens; tokens refill over time to enforce a steady rate."
                ),
                "concepts": ["token bucket", "leaky bucket", "shared store", "refill", "identity"],
                "hint": "Think about tokens, a shared counter, and refill over time."
            }
        ],
        "advanced": [
            {
                "q": "Explain the CAP theorem trade-offs for distributed systems.",
                "answer": (
                    "In the presence of a network partition, you must choose between Consistency and Availability. "
                    "Systems can at most provide any two of Consistency, Availability, and Partition tolerance."
                ),
                "concepts": ["consistency", "availability", "partition tolerance", "trade-off"],
                "hint": "During a partition you make a choice; which two can you keep?"
            }
        ],
    },
    "python": {
        "basic": [
            {
                "q": "What are lists vs tuples in Python, and when use each?",
                "answer": (
                    "Lists are mutable sequences suitable for items that change. "
                    "Tuples are immutable, often used for fixed collections or as dict keys."
                ),
                "concepts": ["mutable", "immutable", "sequence", "use cases"],
                "hint": "One changes, one doesn't; think about when you'd want that."
            },
        ]
    },
}

DEFAULT_TECH = "general"
