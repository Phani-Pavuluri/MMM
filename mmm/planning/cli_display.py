"""CLI helpers: surface planning assumptions and policy warnings on stderr."""

from __future__ import annotations

from typing import Any

import typer


def emit_planning_summary_to_stderr(
    payload: dict[str, Any],
    *,
    echo: Any | None = None,
    secho: Any | None = None,
) -> None:
    """
    Print planning contract fields prominently (assumptions, policy warnings, validation notes).

    ``echo`` / ``secho`` default to ``typer`` helpers.
    """
    echo = echo or typer.echo
    secho = secho or typer.secho
    sim = payload.get("simulation") if isinstance(payload.get("simulation"), dict) else {}
    pa = payload.get("planning_assumptions")
    if not isinstance(pa, dict):
        pa = sim.get("planning_assumptions") if isinstance(sim.get("planning_assumptions"), dict) else None
    if isinstance(pa, dict):
        secho(
            f"Planning assumptions: controls={pa.get('controls_assumption')} "
            f"media={pa.get('media_assumption')} world={pa.get('world_assumption')}",
            fg=typer.colors.CYAN,
        )
        for line in pa.get("planning_disclosures") or []:
            secho(str(line), fg=typer.colors.YELLOW, err=True)

    du = payload.get("decision_uncertainty")
    if not isinstance(du, dict):
        bundle = payload.get("decision_bundle")
        if isinstance(bundle, dict):
            du = bundle.get("decision_uncertainty")
    if isinstance(du, dict) and du.get("disclosure_text"):
        secho(f"Uncertainty: {du.get('disclosure_text')}", fg=typer.colors.YELLOW, err=True)

    pol = payload.get("control_scenario_policy")
    if not isinstance(pol, dict):
        pol = sim.get("control_scenario_policy") if isinstance(sim.get("control_scenario_policy"), dict) else None
    if isinstance(pol, dict):
        sev = str(pol.get("severity", "info"))
        for msg in pol.get("messages") or []:
            if sev == "warning":
                secho(f"PLANNING POLICY WARNING: {msg}", fg=typer.colors.YELLOW, err=True)
            elif sev == "block":
                secho(f"PLANNING POLICY BLOCK: {msg}", fg=typer.colors.RED, err=True)
            else:
                echo(f"Planning policy ({sev}): {msg}")

    warn_sources: list[Any] = [payload]
    sim_block = payload.get("simulation")
    if isinstance(sim_block, dict):
        warn_sources.append(sim_block)
    seen_warn: set[str] = set()
    for block in warn_sources:
        if not isinstance(block, dict):
            continue
        for w in block.get("scenario_validation_warnings") or []:
            ws = str(w)
            if ws in seen_warn:
                continue
            seen_warn.add(ws)
            secho(f"SCENARIO VALIDATION WARNING: {ws}", fg=typer.colors.YELLOW, err=True)

    sl = payload.get("scenario_lineage")
    if isinstance(sl, dict) and sl:
        if sl.get("non_media_overlay_supplied") is False:
            secho(
                "Non-media: no PlanningScenario overlay supplied; using observed historical panel controls.",
                fg=typer.colors.YELLOW,
                err=True,
            )
        elif sl.get("non_media_overlay_applied"):
            secho(
                f"Non-media: fixed scenario applied (scenario_id={sl.get('scenario_id')!r}, "
                f"plan_overlay_sha256={str(sl.get('plan_overlay_spec_sha256') or '')[:12]}…).",
                fg=typer.colors.CYAN,
            )
