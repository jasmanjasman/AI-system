from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Caller:
    """Who is asking, and what they are allowed to do.

    Built by the app from something it trusts — a session, a verified token —
    never from anything the client sends, or a caller could simply declare
    itself an admin. Frozen so nothing downstream can widen it mid-request.
    """

    user_id: str
    permissions: frozenset[str] = field(default_factory=frozenset)

    def has(self, permission: str | None) -> bool:
        """True if this caller may use a tool requiring `permission`.

        None means the tool declared no requirement — a public tool that
        everyone may call. That is the default in ToolSpec, so a provider
        opts INTO restriction rather than having to remember to opt out.
        """
        if permission is None:
            return True
        return permission in self.permissions

    @classmethod
    def anonymous(cls) -> Caller:
        """A caller with no permissions at all: public tools only."""
        return cls(user_id="anonymous")
