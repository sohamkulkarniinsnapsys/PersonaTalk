"""
Domain-agnostic topic taxonomy system for dynamic interview generation.

This module defines the complete topic structure for all supported domains:
- Python, JavaScript, React, Backend Systems, Databases, DevOps, System Design
- Each domain has subtopics categorized by difficulty (beginner, intermediate, advanced)
- All topics are configuration-driven and can be extended without code changes
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class Topic:
    """Represents a single interview topic at a specific difficulty level."""
    name: str  # e.g., "Variables and Types"
    difficulty: str  # "beginner", "intermediate", "advanced"
    description: str  # Brief explanation of what to ask about
    key_concepts: List[str]  # Core concepts (for evaluation hints)
    domain: str = ""  # Parent domain slug (set during taxonomy initialization)


@dataclass
class Subtopic:
    """Represents a domain subtopic with difficulty-stratified topics."""
    name: str  # e.g., "Basic Language Concepts"
    domain: str  # Parent domain (e.g., "python")
    topics: Dict[str, List[Topic]] = field(default_factory=lambda: {
        "beginner": [],
        "intermediate": [],
        "advanced": []
    })


@dataclass
class Domain:
    """Represents a complete interview domain (e.g., Python, JavaScript, React)."""
    name: str  # e.g., "Python"
    slug: str  # "python" (used in config/routing)
    description: str  # "Python programming language"
    subtopics: List[Subtopic] = field(default_factory=list)


class TopicTaxonomy:
    """
    Complete topic taxonomy for all interview domains.
    This is the single source of truth for interview organization.
    """

    def __init__(self):
        self.domains: Dict[str, Domain] = {}
        self._initialize_all_domains()

    def _initialize_all_domains(self):
        """Initialize all predefined domains and their topic hierarchies."""
        self._init_python_domain()
        self._init_javascript_domain()
        self._init_react_domain()
        self._init_backend_domain()
        self._init_databases_domain()
        self._init_devops_domain()
        self._init_system_design_domain()
        self._init_general_domain()
        self._assign_topic_domains()

    def _assign_topic_domains(self):
        """Attach parent domain slug to each Topic for downstream consumers."""
        for domain in self.domains.values():
            for subtopic in domain.subtopics:
                for topic_list in subtopic.topics.values():
                    for topic in topic_list:
                        topic.domain = subtopic.domain

    def _init_python_domain(self):
        """Python programming language domain."""
        python = Domain(
            name="Python",
            slug="python",
            description="Python programming language and ecosystem"
        )

        # Subtopic 1: Basic Language Concepts
        basic_lang = Subtopic(name="Basic Language Concepts", domain="python")
        basic_lang.topics["beginner"] = [
            Topic(
                name="Variables and Types",
                difficulty="beginner",
                description="How to declare variables, type system, int/str/float/bool/list/dict/tuple",
                key_concepts=["variable", "type", "int", "str", "float", "bool", "list", "dict", "tuple"]
            ),
            Topic(
                name="Control Flow",
                difficulty="beginner",
                description="if/elif/else statements, loops (for/while), break/continue",
                key_concepts=["if", "elif", "else", "for", "while", "break", "continue"]
            ),
            Topic(
                name="Lists vs Tuples",
                difficulty="beginner",
                description="Difference between mutable lists and immutable tuples, use cases",
                key_concepts=["mutable", "immutable", "sequence", "use cases", "list", "tuple"]
            ),
            Topic(
                name="String Operations",
                difficulty="beginner",
                description="String concatenation, methods, formatting, f-strings",
                key_concepts=["concatenation", "methods", "formatting", "f-string", "split", "join"]
            ),
        ]

        basic_lang.topics["intermediate"] = [
            Topic(
                name="List Comprehensions",
                difficulty="intermediate",
                description="Creating lists efficiently using comprehension syntax",
                key_concepts=["comprehension", "syntax", "filter", "map", "iteration"]
            ),
            Topic(
                name="Dictionary Operations",
                difficulty="intermediate",
                description="Working with dicts, keys, values, iteration, get/setdefault",
                key_concepts=["dictionary", "keys", "values", "iteration", "get", "setdefault"]
            ),
            Topic(
                name="String Methods and Formatting",
                difficulty="intermediate",
                description="Advanced string operations, regex basics, format methods",
                key_concepts=["methods", "regex", "format", "lower", "upper", "replace", "find"]
            ),
        ]

        basic_lang.topics["advanced"] = [
            Topic(
                name="Generators and Iterators",
                difficulty="advanced",
                description="Generator functions, yield, iterator protocol, lazy evaluation",
                key_concepts=["generator", "yield", "iterator", "lazy evaluation", "__iter__", "__next__"]
            ),
            Topic(
                name="Decorators",
                difficulty="advanced",
                description="Function decorators, wrapping, closure, parameterized decorators",
                key_concepts=["decorator", "wrapping", "closure", "parameterized", "@property"]
            ),
        ]
        python.subtopics.append(basic_lang)

        # Subtopic 2: Object-Oriented Programming
        oop = Subtopic(name="Object-Oriented Programming", domain="python")
        oop.topics["beginner"] = [
            Topic(
                name="Classes and Objects",
                difficulty="beginner",
                description="Class definition, __init__ constructor, instance variables, methods",
                key_concepts=["class", "object", "__init__", "constructor", "instance", "self"]
            ),
            Topic(
                name="Inheritance",
                difficulty="beginner",
                description="Class inheritance, method overriding, super()",
                key_concepts=["inheritance", "subclass", "override", "super"]
            ),
        ]

        oop.topics["intermediate"] = [
            Topic(
                name="Encapsulation and Properties",
                difficulty="intermediate",
                description="Public/private methods, @property decorator, getters/setters",
                key_concepts=["encapsulation", "private", "public", "@property", "getter", "setter"]
            ),
            Topic(
                name="Multiple Inheritance and MRO",
                difficulty="intermediate",
                description="Multiple inheritance, method resolution order (MRO), super() behavior",
                key_concepts=["multiple inheritance", "MRO", "super", "C3 linearization"]
            ),
        ]

        oop.topics["advanced"] = [
            Topic(
                name="Metaclasses and Descriptors",
                difficulty="advanced",
                description="Metaclasses, descriptors, __getattr__, __setattr__",
                key_concepts=["metaclass", "descriptor", "__getattr__", "__setattr__", "__get__", "__set__"]
            ),
        ]
        python.subtopics.append(oop)

        # Subtopic 3: Error Handling and Debugging
        errors = Subtopic(name="Error Handling and Debugging", domain="python")
        errors.topics["beginner"] = [
            Topic(
                name="Exceptions and Try/Except",
                difficulty="beginner",
                description="Exception types, try/except/finally, raise statement",
                key_concepts=["exception", "try", "except", "finally", "raise", "error handling"]
            ),
        ]

        errors.topics["intermediate"] = [
            Topic(
                name="Custom Exceptions",
                difficulty="intermediate",
                description="Creating custom exception classes, exception hierarchy",
                key_concepts=["custom exception", "exception class", "inheritance", "Exception"]
            ),
            Topic(
                name="Context Managers",
                difficulty="intermediate",
                description="with statement, __enter__/__exit__, resource management",
                key_concepts=["context manager", "with", "__enter__", "__exit__", "resource"]
            ),
        ]

        errors.topics["advanced"] = [
            Topic(
                name="Debugging and Profiling",
                difficulty="advanced",
                description="pdb debugger, tracing, profiling, memory leaks",
                key_concepts=["pdb", "debugger", "profiling", "trace", "memory leak"]
            ),
        ]
        python.subtopics.append(errors)

        # Subtopic 4: Standard Library and Modules
        stdlib = Subtopic(name="Standard Library and Modules", domain="python")
        stdlib.topics["beginner"] = [
            Topic(
                name="Importing and Modules",
                difficulty="beginner",
                description="import statement, from/import, module structure, __name__ == '__main__'",
                key_concepts=["import", "module", "from", "package", "__name__", "__main__"]
            ),
        ]

        stdlib.topics["intermediate"] = [
            Topic(
                name="Collections and Itertools",
                difficulty="intermediate",
                description="defaultdict, Counter, namedtuple, itertools functions",
                key_concepts=["defaultdict", "Counter", "namedtuple", "itertools"]
            ),
            Topic(
                name="File and OS Operations",
                difficulty="intermediate",
                description="File I/O, pathlib, os module, working with directories",
                key_concepts=["file I/O", "pathlib", "os", "open", "read", "write"]
            ),
            Topic(
                name="JSON and Data Serialization",
                difficulty="intermediate",
                description="json module, pickle, serialization/deserialization",
                key_concepts=["json", "pickle", "serialize", "deserialize"]
            ),
        ]

        stdlib.topics["advanced"] = [
            Topic(
                name="Concurrency (Threading and Multiprocessing)",
                difficulty="advanced",
                description="threading module, Thread, Lock, multiprocessing, GIL",
                key_concepts=["threading", "multiprocessing", "GIL", "Lock", "Semaphore"]
            ),
        ]
        python.subtopics.append(stdlib)

        # Subtopic 5: Data Structures and Algorithms
        dsa = Subtopic(name="Data Structures and Algorithms", domain="python")
        dsa.topics["intermediate"] = [
            Topic(
                name="Stacks and Queues",
                difficulty="intermediate",
                description="Stack/LIFO, Queue/FIFO, deque, implementation approaches",
                key_concepts=["stack", "LIFO", "queue", "FIFO", "deque", "pop", "append"]
            ),
            Topic(
                name="Linked Lists",
                difficulty="intermediate",
                description="Singly linked lists, doubly linked lists, traversal, insertion, deletion",
                key_concepts=["linked list", "node", "traversal", "insertion", "deletion"]
            ),
        ]

        dsa.topics["advanced"] = [
            Topic(
                name="Trees and Graphs",
                difficulty="advanced",
                description="Binary trees, BST, traversal (DFS/BFS), graphs, adjacency list/matrix",
                key_concepts=["tree", "BST", "DFS", "BFS", "traversal", "graph", "adjacency"]
            ),
            Topic(
                name="Sorting and Search Algorithms",
                difficulty="advanced",
                description="Binary search, quicksort, mergesort, heapsort, time complexity",
                key_concepts=["binary search", "quicksort", "mergesort", "heapsort", "O(n log n)"]
            ),
        ]
        python.subtopics.append(dsa)

        # Subtopic 6: Advanced Topics
        advanced = Subtopic(name="Advanced Topics", domain="python")
        advanced.topics["advanced"] = [
            Topic(
                name="Async and Await",
                difficulty="advanced",
                description="asyncio, async/await, coroutines, event loop, tasks",
                key_concepts=["async", "await", "asyncio", "coroutine", "event loop", "tasks"]
            ),
            Topic(
                name="Memory Management and Performance",
                difficulty="advanced",
                description="Reference counting, garbage collection, memory profiling, optimization",
                key_concepts=["reference counting", "garbage collection", "memory profile", "optimization"]
            ),
            Topic(
                name="Type Hints and Static Analysis",
                difficulty="advanced",
                description="Type hints, mypy, runtime vs static types, generic types",
                key_concepts=["type hint", "mypy", "static analysis", "generic", "Protocol"]
            ),
        ]
        python.subtopics.append(advanced)

        self.domains["python"] = python
        logger.info("✅ Initialized Python domain with 6 subtopics")

    def _init_javascript_domain(self):
        """JavaScript programming language domain."""
        js = Domain(
            name="JavaScript",
            slug="javascript",
            description="JavaScript programming language and runtime"
        )

        # Subtopic 1: Fundamentals
        fundamentals = Subtopic(name="Fundamentals", domain="javascript")
        fundamentals.topics["beginner"] = [
            Topic(
                name="Variables and Scope",
                difficulty="beginner",
                description="var/let/const, function scope, block scope, hoisting",
                key_concepts=["var", "let", "const", "scope", "hoisting", "global", "local"]
            ),
            Topic(
                name="Data Types",
                difficulty="beginner",
                description="Primitive types (number, string, boolean, null, undefined), typeof operator",
                key_concepts=["number", "string", "boolean", "null", "undefined", "typeof"]
            ),
            Topic(
                name="Operators and Expressions",
                difficulty="beginner",
                description="Arithmetic, comparison, logical operators, operator precedence",
                key_concepts=["operator", "precedence", "&&", "||", "!", "==", "==="]
            ),
        ]

        fundamentals.topics["intermediate"] = [
            Topic(
                name="Type Coercion",
                difficulty="intermediate",
                description="Implicit and explicit type conversion, truthy/falsy values",
                key_concepts=["coercion", "truthy", "falsy", "Number()", "String()", "Boolean()"]
            ),
            Topic(
                name="Template Literals",
                difficulty="intermediate",
                description="Template strings, string interpolation, multi-line strings",
                key_concepts=["template literal", "backtick", "interpolation", "${}", "multi-line"]
            ),
        ]

        fundamentals.topics["advanced"] = [
            Topic(
                name="Symbol and Well-known Symbols",
                difficulty="advanced",
                description="Symbol primitive type, Symbol.iterator, Symbol.hasInstance",
                key_concepts=["Symbol", "well-known symbols", "Symbol.iterator", "unique"]
            ),
        ]
        js.subtopics.append(fundamentals)

        # Subtopic 2: Functions and Closures
        functions = Subtopic(name="Functions and Closures", domain="javascript")
        functions.topics["beginner"] = [
            Topic(
                name="Function Declaration and Expression",
                difficulty="beginner",
                description="Function declaration, function expression, arrow functions, parameters",
                key_concepts=["function declaration", "expression", "arrow", "=>", "parameters"]
            ),
            Topic(
                name="Return Statements",
                difficulty="beginner",
                description="Return values, multiple returns, returning functions",
                key_concepts=["return", "undefined", "void"]
            ),
        ]

        functions.topics["intermediate"] = [
            Topic(
                name="Closures",
                difficulty="intermediate",
                description="Closure concept, lexical scope, function factory, private variables",
                key_concepts=["closure", "lexical scope", "outer function", "private variable"]
            ),
            Topic(
                name="Higher-Order Functions",
                difficulty="intermediate",
                description="Functions as arguments, returning functions, map/filter/reduce",
                key_concepts=["higher-order", "callback", "map", "filter", "reduce"]
            ),
            Topic(
                name="Function Context (this)",
                difficulty="intermediate",
                description="this binding, call, apply, bind methods",
                key_concepts=["this", "call", "apply", "bind", "context"]
            ),
        ]

        functions.topics["advanced"] = [
            Topic(
                name="Currying and Partial Application",
                difficulty="advanced",
                description="Currying, partial application, function composition",
                key_concepts=["currying", "partial application", "composition"]
            ),
        ]
        js.subtopics.append(functions)

        # Subtopic 3: Objects and Prototypes
        objects = Subtopic(name="Objects and Prototypes", domain="javascript")
        objects.topics["beginner"] = [
            Topic(
                name="Object Literals and Properties",
                difficulty="beginner",
                description="Object literal syntax, property access (dot/bracket), adding/deleting properties",
                key_concepts=["object literal", "property", "dot notation", "bracket notation"]
            ),
            Topic(
                name="Objects vs Primitives",
                difficulty="beginner",
                description="Reference vs value types, object mutability",
                key_concepts=["reference type", "value type", "mutability", "copy"]
            ),
        ]

        objects.topics["intermediate"] = [
            Topic(
                name="Prototype Chain",
                difficulty="intermediate",
                description="Prototype inheritance, __proto__, prototype property, Object.create()",
                key_concepts=["prototype", "__proto__", "prototype chain", "Object.create"]
            ),
            Topic(
                name="Classes and Constructors",
                difficulty="intermediate",
                description="Constructor functions, new keyword, class syntax (ES6), instanceof",
                key_concepts=["constructor", "new", "class", "ES6", "instanceof"]
            ),
        ]

        objects.topics["advanced"] = [
            Topic(
                name="Property Descriptors and Getters/Setters",
                difficulty="advanced",
                description="Object.defineProperty, property descriptors, getters, setters",
                key_concepts=["property descriptor", "defineProperty", "getter", "setter", "configurable"]
            ),
        ]
        js.subtopics.append(objects)

        # Subtopic 4: Async Programming
        async_prog = Subtopic(name="Async Programming", domain="javascript")
        async_prog.topics["intermediate"] = [
            Topic(
                name="Callbacks",
                difficulty="intermediate",
                description="Callback functions, callback hell, error handling in callbacks",
                key_concepts=["callback", "callback hell", "nested callbacks"]
            ),
            Topic(
                name="Promises",
                difficulty="intermediate",
                description="Promise states (pending/fulfilled/rejected), .then/.catch/.finally, chaining",
                key_concepts=["promise", "pending", "fulfilled", "rejected", "then", "catch"]
            ),
        ]

        async_prog.topics["advanced"] = [
            Topic(
                name="Async/Await",
                difficulty="advanced",
                description="async function, await expression, error handling with try/catch",
                key_concepts=["async", "await", "try/catch", "error handling"]
            ),
            Topic(
                name="Event Loop and Microtask Queue",
                difficulty="advanced",
                description="Call stack, task queue, microtask queue, execution order",
                key_concepts=["event loop", "call stack", "microtask", "macrotask", "order"]
            ),
        ]
        js.subtopics.append(async_prog)

        # Subtopic 5: DOM and Browser APIs
        dom = Subtopic(name="DOM and Browser APIs", domain="javascript")
        dom.topics["intermediate"] = [
            Topic(
                name="DOM Manipulation",
                difficulty="intermediate",
                description="querySelector, getElementById, createElement, appendChild, innerHTML",
                key_concepts=["DOM", "querySelector", "getElementById", "createElement", "appendChild"]
            ),
            Topic(
                name="Event Handling",
                difficulty="intermediate",
                description="addEventListener, event object, event propagation, delegation",
                key_concepts=["addEventListener", "event", "propagation", "bubbling", "capturing"]
            ),
        ]

        dom.topics["advanced"] = [
            Topic(
                name="LocalStorage and IndexedDB",
                difficulty="advanced",
                description="localStorage, sessionStorage, IndexedDB for persistent storage",
                key_concepts=["localStorage", "sessionStorage", "IndexedDB", "persistence"]
            ),
        ]
        js.subtopics.append(dom)

        self.domains["javascript"] = js
        logger.info("✅ Initialized JavaScript domain with 5 subtopics")

    def _init_react_domain(self):
        """React framework domain."""
        react = Domain(
            name="React",
            slug="react",
            description="React library for building user interfaces"
        )

        # Subtopic 1: Fundamentals
        fundamentals = Subtopic(name="Fundamentals", domain="react")
        fundamentals.topics["beginner"] = [
            Topic(
                name="Components and JSX",
                difficulty="beginner",
                description="Functional components, class components, JSX syntax",
                key_concepts=["component", "JSX", "functional", "class", "render"]
            ),
            Topic(
                name="Props",
                difficulty="beginner",
                description="Props definition, passing props, prop types, default props",
                key_concepts=["props", "prop drilling", "PropTypes", "defaultProps"]
            ),
            Topic(
                name="State and useState",
                difficulty="beginner",
                description="State concept, useState hook, updating state, state immutability",
                key_concepts=["state", "useState", "setState", "immutability"]
            ),
        ]

        fundamentals.topics["intermediate"] = [
            Topic(
                name="Event Handling",
                difficulty="intermediate",
                description="Event handlers, synthetic events, event binding, preventDefault",
                key_concepts=["event handler", "synthetic event", "binding", "preventDefault"]
            ),
            Topic(
                name="Conditional Rendering",
                difficulty="intermediate",
                description="if/else in render, ternary operator, logical &&, switch patterns",
                key_concepts=["conditional", "ternary", "&&", "if", "else"]
            ),
        ]

        fundamentals.topics["advanced"] = [
            Topic(
                name="Render Props and Composition",
                difficulty="advanced",
                description="Render props pattern, component composition, children as function",
                key_concepts=["render props", "composition", "children", "function as child"]
            ),
        ]
        react.subtopics.append(fundamentals)

        # Subtopic 2: Hooks
        hooks = Subtopic(name="Hooks", domain="react")
        hooks.topics["beginner"] = [
            Topic(
                name="useState Hook",
                difficulty="beginner",
                description="useState basic usage, updating state, initial state",
                key_concepts=["useState", "state variable", "setState", "initial"]
            ),
        ]

        hooks.topics["intermediate"] = [
            Topic(
                name="useEffect Hook",
                difficulty="intermediate",
                description="useEffect lifecycle, dependency array, cleanup function, side effects",
                key_concepts=["useEffect", "lifecycle", "dependencies", "cleanup", "side effect"]
            ),
            Topic(
                name="useContext Hook",
                difficulty="intermediate",
                description="Context API, useContext hook, prop drilling avoidance",
                key_concepts=["useContext", "Context", "Provider", "Consumer", "prop drilling"]
            ),
            Topic(
                name="useReducer Hook",
                difficulty="intermediate",
                description="useReducer for complex state, reducer function, dispatch",
                key_concepts=["useReducer", "reducer", "action", "dispatch", "complex state"]
            ),
        ]

        hooks.topics["advanced"] = [
            Topic(
                name="Custom Hooks",
                difficulty="advanced",
                description="Creating custom hooks, hook rules, useCallback, useMemo",
                key_concepts=["custom hook", "useCallback", "useMemo", "hook rules"]
            ),
            Topic(
                name="Advanced Hooks",
                difficulty="advanced",
                description="useRef, useLayoutEffect, useId, useDeferredValue, useTransition",
                key_concepts=["useRef", "useLayoutEffect", "useId", "concurrency"]
            ),
        ]
        react.subtopics.append(hooks)

        # Subtopic 3: Component Lifecycle and Performance
        perf = Subtopic(name="Component Lifecycle and Performance", domain="react")
        perf.topics["intermediate"] = [
            Topic(
                name="Component Lifecycle (Class Components)",
                difficulty="intermediate",
                description="Lifecycle methods, mounting, updating, unmounting, getDerivedStateFromProps",
                key_concepts=["lifecycle", "componentDidMount", "componentDidUpdate", "componentWillUnmount"]
            ),
            Topic(
                name="Optimization Strategies",
                difficulty="intermediate",
                description="useMemo, useCallback, React.memo, key prop optimization",
                key_concepts=["optimization", "useMemo", "useCallback", "React.memo", "key"]
            ),
        ]

        perf.topics["advanced"] = [
            Topic(
                name="Reconciliation and Virtual DOM",
                difficulty="advanced",
                description="React reconciliation algorithm, Virtual DOM, fiber architecture",
                key_concepts=["reconciliation", "virtual DOM", "fiber", "diffing"]
            ),
            Topic(
                name="Concurrent Features",
                difficulty="advanced",
                description="Concurrent rendering, Suspense, automatic batching, startTransition",
                key_concepts=["concurrent", "Suspense", "batching", "startTransition"]
            ),
        ]
        react.subtopics.append(perf)

        # Subtopic 4: State Management
        state = Subtopic(name="State Management", domain="react")
        state.topics["intermediate"] = [
            Topic(
                name="Local Component State",
                difficulty="intermediate",
                description="useState best practices, lifting state up, controlled components",
                key_concepts=["local state", "lifting state", "controlled component"]
            ),
            Topic(
                name="Global State Patterns",
                difficulty="intermediate",
                description="Context + useReducer, custom hooks for state, state machines",
                key_concepts=["global state", "Context", "custom hook", "state machine"]
            ),
        ]

        state.topics["advanced"] = [
            Topic(
                name="Redux and External State",
                difficulty="advanced",
                description="Redux store, actions, reducers, middleware, Redux Toolkit",
                key_concepts=["Redux", "store", "action", "reducer", "middleware"]
            ),
        ]
        react.subtopics.append(state)

        self.domains["react"] = react
        logger.info("✅ Initialized React domain with 4 subtopics")

    def _init_backend_domain(self):
        """Backend systems and web development domain."""
        backend = Domain(
            name="Backend Systems",
            slug="backend",
            description="Backend systems, APIs, and web server fundamentals"
        )

        # Subtopic 1: API Design
        api = Subtopic(name="API Design", domain="backend")
        api.topics["beginner"] = [
            Topic(
                name="REST API Basics",
                difficulty="beginner",
                description="RESTful principles, HTTP verbs, resource-oriented design",
                key_concepts=["REST", "HTTP verb", "GET", "POST", "PUT", "DELETE", "resource"]
            ),
            Topic(
                name="HTTP Status Codes",
                difficulty="beginner",
                description="2xx/3xx/4xx/5xx codes, common codes (200, 201, 400, 404, 500)",
                key_concepts=["status code", "2xx", "4xx", "5xx", "success", "error"]
            ),
        ]

        api.topics["intermediate"] = [
            Topic(
                name="API Versioning and Headers",
                difficulty="intermediate",
                description="API versioning strategies, headers, content negotiation, CORS",
                key_concepts=["versioning", "headers", "content-type", "CORS", "accept"]
            ),
            Topic(
                name="Authentication and Authorization",
                difficulty="intermediate",
                description="Basic auth, JWT, OAuth, session tokens, permission scopes",
                key_concepts=["authentication", "JWT", "OAuth", "session", "scope", "permission"]
            ),
        ]

        api.topics["advanced"] = [
            Topic(
                name="GraphQL Basics",
                difficulty="advanced",
                description="GraphQL vs REST, queries, mutations, schema, resolvers",
                key_concepts=["GraphQL", "query", "mutation", "schema", "resolver"]
            ),
        ]
        backend.subtopics.append(api)

        # Subtopic 2: Web Server Fundamentals
        server = Subtopic(name="Web Server Fundamentals", domain="backend")
        server.topics["intermediate"] = [
            Topic(
                name="Request/Response Cycle",
                difficulty="intermediate",
                description="HTTP request, response headers, body, middleware pipeline",
                key_concepts=["request", "response", "headers", "body", "middleware"]
            ),
            Topic(
                name="Routing",
                difficulty="intermediate",
                description="Route definition, path parameters, query strings, route matching",
                key_concepts=["routing", "path parameter", "query string", "route"]
            ),
        ]

        server.topics["advanced"] = [
            Topic(
                name="Middleware and Interceptors",
                difficulty="advanced",
                description="Middleware pattern, request/response interceptors, error handling",
                key_concepts=["middleware", "interceptor", "error handling", "pipeline"]
            ),
        ]
        backend.subtopics.append(server)

        # Subtopic 3: Database Integration
        db = Subtopic(name="Database Integration", domain="backend")
        db.topics["intermediate"] = [
            Topic(
                name="ORM Concepts",
                difficulty="intermediate",
                description="Object-Relational Mapping, models, migrations, relationships",
                key_concepts=["ORM", "model", "migration", "relationship", "foreign key"]
            ),
        ]

        db.topics["advanced"] = [
            Topic(
                name="N+1 Queries and Optimization",
                difficulty="advanced",
                description="N+1 problem, eager loading, lazy loading, query optimization",
                key_concepts=["N+1", "eager loading", "lazy loading", "optimization"]
            ),
        ]
        backend.subtopics.append(db)

        self.domains["backend"] = backend
        logger.info("✅ Initialized Backend domain with 3 subtopics")

    def _init_databases_domain(self):
        """Databases domain."""
        db = Domain(
            name="Databases",
            slug="databases",
            description="SQL and NoSQL database systems"
        )

        # Subtopic 1: SQL Basics
        sql = Subtopic(name="SQL Basics", domain="databases")
        sql.topics["beginner"] = [
            Topic(
                name="SELECT and WHERE",
                difficulty="beginner",
                description="Basic SELECT queries, WHERE conditions, AND/OR/NOT",
                key_concepts=["SELECT", "WHERE", "AND", "OR", "NOT", "column"]
            ),
            Topic(
                name="INSERT, UPDATE, DELETE",
                difficulty="beginner",
                description="Data manipulation, INSERT values, UPDATE rows, DELETE rows",
                key_concepts=["INSERT", "UPDATE", "DELETE", "VALUES", "SET"]
            ),
        ]

        sql.topics["intermediate"] = [
            Topic(
                name="JOINs and Relationships",
                difficulty="intermediate",
                description="INNER JOIN, LEFT JOIN, RIGHT JOIN, FULL JOIN, cross join",
                key_concepts=["JOIN", "INNER", "LEFT", "RIGHT", "FULL", "foreign key"]
            ),
            Topic(
                name="Aggregation and Grouping",
                difficulty="intermediate",
                description="GROUP BY, HAVING, COUNT, SUM, AVG, MIN, MAX",
                key_concepts=["GROUP BY", "HAVING", "COUNT", "SUM", "AVG", "aggregate"]
            ),
        ]

        sql.topics["advanced"] = [
            Topic(
                name="Indexes and Query Optimization",
                difficulty="advanced",
                description="Index types, EXPLAIN PLAN, query optimization, performance tuning",
                key_concepts=["index", "EXPLAIN", "optimization", "primary key", "composite"]
            ),
            Topic(
                name="Transactions and ACID",
                difficulty="advanced",
                description="ACID properties, BEGIN/COMMIT/ROLLBACK, isolation levels",
                key_concepts=["transaction", "ACID", "isolation", "consistency"]
            ),
        ]
        db.subtopics.append(sql)

        # Subtopic 2: NoSQL Basics
        nosql = Subtopic(name="NoSQL Basics", domain="databases")
        nosql.topics["intermediate"] = [
            Topic(
                name="Document Databases",
                difficulty="intermediate",
                description="MongoDB, documents, collections, BSON, schema flexibility",
                key_concepts=["document", "MongoDB", "collection", "BSON", "schema-less"]
            ),
            Topic(
                name="Key-Value Stores",
                difficulty="intermediate",
                description="Redis, key-value pairs, data types (string, list, set, hash)",
                key_concepts=["key-value", "Redis", "string", "list", "set", "hash"]
            ),
        ]

        nosql.topics["advanced"] = [
            Topic(
                name="Scaling and Replication",
                difficulty="advanced",
                description="Sharding, replication, CAP theorem, eventual consistency",
                key_concepts=["sharding", "replication", "CAP", "consistency", "availability"]
            ),
        ]
        db.subtopics.append(nosql)

        self.domains["databases"] = db
        logger.info("✅ Initialized Databases domain with 2 subtopics")

    def _init_devops_domain(self):
        """DevOps domain."""
        devops = Domain(
            name="DevOps",
            slug="devops",
            description="DevOps practices, CI/CD, deployment, and infrastructure"
        )

        # Subtopic 1: Containerization
        containers = Subtopic(name="Containerization", domain="devops")
        containers.topics["beginner"] = [
            Topic(
                name="Docker Basics",
                difficulty="beginner",
                description="Docker images, containers, Dockerfile, docker run/build commands",
                key_concepts=["Docker", "image", "container", "Dockerfile", "build"]
            ),
        ]

        containers.topics["intermediate"] = [
            Topic(
                name="Docker Compose",
                difficulty="intermediate",
                description="Multi-container orchestration, docker-compose.yml, services",
                key_concepts=["docker-compose", "service", "network", "volume"]
            ),
        ]

        containers.topics["advanced"] = [
            Topic(
                name="Kubernetes Basics",
                difficulty="advanced",
                description="Pods, deployments, services, ingress, scaling",
                key_concepts=["Kubernetes", "pod", "deployment", "service", "ingress"]
            ),
        ]
        devops.subtopics.append(containers)

        # Subtopic 2: CI/CD
        cicd = Subtopic(name="CI/CD", domain="devops")
        cicd.topics["intermediate"] = [
            Topic(
                name="Continuous Integration",
                difficulty="intermediate",
                description="Automated testing, build pipelines, linting, code quality",
                key_concepts=["CI", "testing", "build", "linting", "quality gate"]
            ),
            Topic(
                name="Continuous Deployment",
                difficulty="intermediate",
                description="Deployment pipelines, staging/production, blue-green deployment",
                key_concepts=["CD", "deployment", "staging", "production", "blue-green"]
            ),
        ]

        cicd.topics["advanced"] = [
            Topic(
                name="Advanced CI/CD Patterns",
                difficulty="advanced",
                description="Canary releases, feature flags, rollback strategies, monitoring",
                key_concepts=["canary", "feature flag", "rollback", "monitoring"]
            ),
        ]
        devops.subtopics.append(cicd)

        self.domains["devops"] = devops
        logger.info("✅ Initialized DevOps domain with 2 subtopics")

    def _init_system_design_domain(self):
        """System Design domain."""
        sd = Domain(
            name="System Design",
            slug="system-design",
            description="Large-scale system design and architecture"
        )

        # Subtopic 1: Core Concepts
        core = Subtopic(name="Core Concepts", domain="system-design")
        core.topics["intermediate"] = [
            Topic(
                name="Scalability and Load Balancing",
                difficulty="intermediate",
                description="Horizontal/vertical scaling, load balancers, traffic distribution",
                key_concepts=["scalability", "load balancer", "horizontal", "vertical", "traffic"]
            ),
            Topic(
                name="Caching Strategies",
                difficulty="intermediate",
                description="Cache invalidation, cache-aside, write-through, TTL",
                key_concepts=["caching", "invalidation", "cache-aside", "TTL", "hit ratio"]
            ),
        ]

        core.topics["advanced"] = [
            Topic(
                name="CAP Theorem and Consistency",
                difficulty="advanced",
                description="CAP theorem trade-offs, consistency models, eventual consistency",
                key_concepts=["CAP", "consistency", "availability", "partition", "eventual"]
            ),
            Topic(
                name="Distributed System Patterns",
                difficulty="advanced",
                description="Service discovery, circuit breaker, retry logic, fallback",
                key_concepts=["service discovery", "circuit breaker", "retry", "fallback"]
            ),
        ]
        sd.subtopics.append(core)

        self.domains["system-design"] = sd
        logger.info("✅ Initialized System Design domain with 1 subtopic")

    def _init_general_domain(self):
        """General technical interview domain (fallback)."""
        general = Domain(
            name="General",
            slug="general",
            description="General technical concepts and fundamentals"
        )

        # Subtopic 1: Fundamentals
        fundamentals = Subtopic(name="Fundamentals", domain="general")
        fundamentals.topics["beginner"] = [
            Topic(
                name="REST API Basics",
                difficulty="beginner",
                description="RESTful principles, HTTP verbs, status codes",
                key_concepts=["REST", "HTTP", "GET", "POST", "PUT", "DELETE"]
            ),
            Topic(
                name="Basic Algorithms",
                difficulty="beginner",
                description="Time complexity, space complexity, Big O notation",
                key_concepts=["algorithm", "time complexity", "space complexity", "Big O"]
            ),
        ]

        fundamentals.topics["intermediate"] = [
            Topic(
                name="Data Structures",
                difficulty="intermediate",
                description="Arrays, lists, stacks, queues, hash tables, trees",
                key_concepts=["data structure", "array", "list", "stack", "queue", "tree"]
            ),
        ]

        fundamentals.topics["advanced"] = [
            Topic(
                name="Distributed Systems",
                difficulty="advanced",
                description="CAP theorem, consistency, availability, partition tolerance",
                key_concepts=["distributed", "CAP", "consistency", "availability"]
            ),
        ]
        general.subtopics.append(fundamentals)

        self.domains["general"] = general
        logger.info("✅ Initialized General domain with 1 subtopic")

    def get_domain(self, slug: str) -> Domain | None:
        """Get a domain by slug."""
        return self.domains.get(slug)

    def get_all_domains(self) -> List[Domain]:
        """Get all available domains."""
        return list(self.domains.values())

    def get_topics_for_domain_difficulty(
        self,
        domain_slug: str,
        difficulty: str
    ) -> List[Topic]:
        """Get all topics for a domain at a specific difficulty level."""
        domain = self.get_domain(domain_slug)
        if not domain:
            return []
        
        topics = []
        for subtopic in domain.subtopics:
            topics.extend(subtopic.topics.get(difficulty, []))
        return topics

    def get_difficulty_distribution(self) -> Dict[str, int]:
        """Standard difficulty distribution for interviews: 4 beginner, 3 intermediate, 3 advanced."""
        return {
            "beginner": 4,
            "intermediate": 3,
            "advanced": 3
        }

    def list_domains(self) -> List[Dict[str, str]]:
        """List all available domains for display."""
        return [
            {"slug": d.slug, "name": d.name, "description": d.description}
            for d in self.get_all_domains()
        ]


# Global singleton instance
_taxonomy = None


def get_taxonomy() -> TopicTaxonomy:
    """Get the global topic taxonomy instance."""
    global _taxonomy
    if _taxonomy is None:
        _taxonomy = TopicTaxonomy()
    return _taxonomy
