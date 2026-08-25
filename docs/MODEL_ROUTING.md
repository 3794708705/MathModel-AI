# Model routing

Agents submit an explicit `TaskProfile`. `ModelRouter` combines profile demand,
task minimums, deadline pressure, blast radius, and retry count, then returns a
recordable `RouteDecision`.

Logical escalation is:

```text
FAST -> BALANCED -> FLAGSHIP_HIGH -> FLAGSHIP_XHIGH -> FLAGSHIP_MAX
     -> MULTI_MODEL_REVIEW -> HUMAN_REVIEW
```

Mathematical modeling starts at `FLAGSHIP_HIGH`, sandbox security starts at
`FLAGSHIP_XHIGH`, and documentation may start at `FAST`. Problem understanding
starts at `FLAGSHIP_HIGH`; model exploration and jury start at
`FLAGSHIP_XHIGH`. Data understanding starts at `FLAGSHIP_HIGH`; multimodal or
large-context data tasks route at `FLAGSHIP_XHIGH` through the configured Google
target (`gemini-3.7-flash` by default). Model identifiers live in settings,
never in agents. The
logical `FLAGSHIP_MAX` level currently maps to the provider's documented
`xhigh` API effort while retaining `MAX` in audit metadata; the API does not
receive an invented `max` value.

Multi-model and human review are actions, not fake provider calls. The current
environment continues with its configured provider when switching is not
available, while preserving `recommended_model` and `recommended_reasoning`.
