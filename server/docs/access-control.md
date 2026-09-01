# Access Control — Scan Quota

> ⚠️ **Being superseded by [billing.md](billing.md).** The `scansRemaining`
> counter described here is now the FALLBACK path, used when no credit ledger is
> configured (`DATABASE_URL` unset) or when the ledger is unreachable. Everything
> below is still accurate for that path.
>
> Two changes have already landed that this document does not describe:
> - The broker session token no longer snapshots `scansRemaining` — a 12-hour
>   token carrying a quota was a 12-hour grant.
> - `_require_scan_access` reads the live ledger when `BILLING_ENFORCE` is set,
>   and fails open on any ledger problem. It is advisory; the authoritative gate
>   is the credit hold taken at each dispatch point.
>
> `tier: "approved"` is NOT going away — it maps to `credit_accounts.unlimited`
> and stays as break-glass.

> How a user is granted access to a SLAM session, and how to manage it.
> Replaces the old binary `approved` tier with a per-user **scan quota**. Touches both frontends ([frontend.md](frontend.md)) and the worker auth path ([streaming-server.md](streaming-server.md)); the non-obvious traps are in [gotchas.md](gotchas.md).

## Model

Access is a **count of scans remaining**, stored in Clerk at `user.publicMetadata.scansRemaining` (a number).

- **New users get 2 free scans.** Seeding is *just-in-time*: a missing `scansRemaining` is treated as the default (`2`) everywhere, and the value is only written to Clerk on first use. So a user who has never scanned has **no `scansRemaining` field at all** yet still has 2.
- **Both live scans and demo videos consume one scan.** The deduction happens **when tracking starts** — after the GPU worker session is created, just before navigating to the planner.
- **`tier === "approved"` is an unlimited bypass.** Approved accounts never spend a scan (admins / early users). This is the only remaining use of the old `tier` field.
- The default lives in one place per layer: `DEFAULT_SCANS` (`landing/app/utils/scans.ts` and `server/app.py`).

## Where it is enforced

Two layers, because the worker only ever sees a (cacheable, lagging) JWT claim:

1. **Authoritative — landing API route** `landing/app/api/scans/consume/route.ts` (`POST /api/scans/consume`). Reads the **live** balance from Clerk via `clerkClient().users.getUser`, then:
   - approved → `{ remaining: null, unlimited: true }`, spends nothing;
   - `remaining <= 0` → **402** `{ error: "no_scans_remaining", remaining: 0 }`;
   - otherwise decrements via `updateUserMetadata` and returns `{ remaining }`.
2. **Best-effort — worker gate** `server/app.py` `_require_scan_access()` / `_scans_remaining()`. Reads the `scansRemaining` **JWT claim** and rejects with `403 no_scans_remaining` when it is present and `<= 0`. An absent claim defaults to `2` (permissive), so this gate only bites once the Clerk JWT template carries the claim (see Management). Enforced on `/auth/session`, `/auth/qr-token`, every protected HTTP route, and the Socket.IO connect.

> **Test-only bypass.** `DANGEROUSLY_DISABLE_AUTH=1` (env, read in `server/app.py`) short-circuits **both** `_verify_http_token` and `_verify_socket_auth` to return synthetic `tier=approved` claims — disabling all token verification *and* the scan gate (approved ⇒ unlimited). For throwaway test deployments only (open the Modal URL directly, no Clerk/landing); OFF by default, prints a loud boot banner, and **must never be set on the production app**. See [modal-deployment.md](modal-deployment.md) → *Throwaway test deployments*.

> **Durable session token & the claim.** The broker can mint its own HS256 **session token** (`_issue_session_token` → `POST /api/session/refresh`, used by the revisit page to outlive the short-lived hash JWT — see [gotchas.md](gotchas.md)). It **snapshots** `tier` + `scansRemaining` from the bootstrapping Clerk JWT, so `_require_scan_access` keeps gating revisit reads/Q&A consistently. It is identity + a quota *snapshot*, **never a grant** — the authoritative balance still lives in Clerk and is only spent by the landing `consume` route. `_verify_any_token` accepts it on the HTTP path; the Socket.IO connect stays Clerk-only.

**Ordering matters:** the client spends the scan **after** `createModalSession` succeeds, never before — because the JWT claim lags the live balance by the Clerk token-cache TTL, deducting first could 403 a user's legitimate *last* scan. See `launchScan` (`landing/app/dashboard/page.tsx`) and `navigateToPlan` (`landing/app/utils/navigation.ts`), both of which call `consumeScan()` between session creation and navigation. A 402/403 surfaces as `OutOfScansError` and triggers `user.reload()` so the UI re-renders into the out-of-scans state.

## What the user sees

The dashboard (`landing/app/dashboard/page.tsx`) shows the balance under the launch button — "*N free scans remaining*", or "*Unlimited scans on this account*" for approved users. When the balance hits 0 (and not approved) the old waitlist screen is replaced by an **"out of scans"** screen.

## Files

| File | Role |
|------|------|
| `landing/app/utils/scans.ts` | `scansRemaining()` (number, or `null` = unlimited), `hasScanAccess()`, `DEFAULT_SCANS` — the shared gate helper |
| `landing/app/api/scans/consume/route.ts` | Authoritative spend endpoint (`POST /api/scans/consume`) |
| `landing/app/dashboard/page.tsx` | Gate + balance display; spends a scan on live launch |
| `landing/app/utils/navigation.ts` | `consumeScan()`, `OutOfScansError`; spends a scan on the demo path |
| `landing/app/components/DemoCarousel.tsx`, `landing/app/onboarding/HelpButtonMount.tsx` | Gate the demo carousel + tour copy via `hasScanAccess()` |
| `server/app.py` | `_require_scan_access()` / `_scans_remaining()` worker gate |
| `landing/app/api/scans/consume/route.test.ts`, `tests/test_scan_access.py` | Tests |

## Management

### Required Clerk setup (one-time)

The worker gate (layer 2) only enforces once the JWT carries the claim. In the **Clerk Dashboard → JWT Templates → `modal-slam`**, add (keep the existing `tier` claim):

```json
"scansRemaining": "{{user.public_metadata.scansRemaining}}"
```

Until this is added, only the landing `/api/scans/consume` route enforces the quota (which still covers every launch); the worker stays permissive. See the matching trap in [gotchas.md](gotchas.md).

### View / change a single user's balance

**Clerk Dashboard → Users → [user] → Metadata → Public.** `scansRemaining` appears there.
- **Top up:** set `scansRemaining` to any number.
- **Reset to default:** delete the `scansRemaining` field (absent ⇒ 2).
- **Grant unlimited:** set `tier` to `"approved"`.

> Heads-up: a user who has never scanned shows **no `scansRemaining` field** (just-in-time seeding writes it on first use). Blank ≠ zero — they still have 2.

### Not yet built (options if you want them)

- **Seed `scansRemaining: 2` at signup** via a Clerk `user.created` webhook + API route, so every account shows the field explicitly from the start (instead of blank-until-first-use).
- **In-app admin list** of all users + their remaining scans via `clerkClient().users.getUserList()` (the Clerk dashboard only shows one user at a time).
