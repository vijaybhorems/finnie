"""Long-term memory: durable facts a user shares in chat, recalled later.

  extractor.py  decides whether a message may hold facts, asks the model for
                them (structured output), and drops anything sensitive.
  service.py    remember_turn(): extract, filter, save — run in the background
                after a chat turn so it never delays the answer.

Storage and recall live on UserData (src/persistence/user_data.py), the single
path to a user's data; the hydrate node injects the few memories relevant to
each question into the agent's per-request context.
"""
