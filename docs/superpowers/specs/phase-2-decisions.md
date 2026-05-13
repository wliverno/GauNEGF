# Phase II Step 2 Decisions

Date: 2026-05-10
Source: User answers via AskUserQuestion in Phase II Task 3.

| Module | Decision | Notes |
|--------|----------|-------|
| surfG3D | Include in main API (Contact Models > 3D Contacts subsection) | Default per spec. |
| spinTools | Include in main API (Utilities > Spin Tools subsection) | Default per spec. |
| config | Include as Configuration Reference (top-level section after API Reference) | Default per spec. |
| protocols | Include as Developer / Extensibility Reference (top-level section after Configuration Reference) | Default per spec. Distinguishes contributor interface from user-facing API. |
| utils | Include in main API (Utilities > JIT/Linear Algebra Helpers subsection) | Flipped from spec default per Phase I usage scan (examples/SiNEGF.py + 3 tests). |
| fermiSearch | Exclude from API docs | Default per spec. Module docstring still gets a clear "DEPRECATED -- use density.calcFermi*" pointer for in-source readers, but no Sphinx page. |

Drives Task 9 (API index update).
