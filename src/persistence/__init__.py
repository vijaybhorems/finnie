"""Per-user persistence: profile, holdings, saved plan, and chat history.

  backend.py    LangGraph checkpointer (chat threads) and store (user data):
                Postgres when DATABASE_URL is set, in-process memory otherwise.
  identity.py   Stable user id from the authenticated Google identity.
  user_data.py  UserData — the only way to read or write a user's data, bound
                to one user id at construction.

Isolation between users is enforced in the application, not by database
row-level security: LangGraph owns the checkpoint and store tables and their
queries, so there is no per-request hook to set a session variable for a
policy. Instead, every read and write goes through a namespace or thread id
derived server-side from the authenticated identity — never from anything the
browser sends — and tests/test_persistence*.py pin that no path crosses users.
"""
