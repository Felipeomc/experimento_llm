#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_llm_stfp.py — Experimento 6: avaliação de equipes recomendadas por LLM
================================================================================

Objetivo:
  - Avaliar, com o mesmo avaliador surrogate do GA/ILP, as equipes recomendadas
    manualmente por um LLM para os projetos P1-P12.
  - Salvar um CSV de execuções e um CSV de resumo compatíveis com o bench do GA.

Uso recomendado, colocando este arquivo no mesmo diretório do bench_ga_stfp.py:

  python bench_llm_stfp.py --restart

Se a raiz do projeto STFP não for detectada automaticamente:

  python bench_llm_stfp.py --project-root D:\\DOCUMENTS\\works\\STFP --restart

Se quiser usar um JSON externo com as equipes do LLM:

  python bench_llm_stfp.py --llm-teams llm_teams_surrogate_input.json --restart

Observação metodológica:
  - O campo tempo_s usa, por padrão, o tempo total retornado pelo chat dividido
    pelo número de projetos avaliados. Como o chat retornou apenas tempo total,
    esse valor deve ser reportado como aproximação por projeto.
  - O tempo real gasto pelo avaliador surrogate fica em tempo_eval_surrogate_s.
"""

from __future__ import annotations

# =============================================================================
# CONSTANTES CONFIGURÁVEIS
# =============================================================================

CAMINHO_PROJETOS = "target_projects.json"
CAMINHO_SAIDA_RUNS = "resultados_llm_surrogate_12projetos_runs.csv"
CAMINHO_SAIDA_RESUMO = "resultados_llm_surrogate_12projetos_resumo.csv"

MODO = "AE_FULL_SURROGATE"
METODO_DEFAULT = "LLM_CHAT_MANUAL"
HARD_MUST_PENALTY = -1.0

# Tempo total retornado pelo chat: "Pensou por 6m 47s" = 407 segundos.
LLM_TOTAL_TIME_S_DEFAULT = 407.0

PROJETOS_IDS = [
    "P1", "P2", "P3", "P4", "P5", "P6",
    "P7", "P8", "P9", "P10", "P11", "P12",
]

# Equipes recomendadas pelo LLM, já convertidas de Dev_XXX para inteiros.
# Fonte: resposta manual do LLM para o Experimento 6.
EQUIPES_LLM_DEFAULT = {
    "P1":  [211, 81, 145, 202],
    "P2":  [490, 480, 311, 174],
    "P3":  [465, 638, 496, 466],
    "P4":  [503, 359, 492, 351],
    "P5":  [65, 181, 351, 631],
    "P6":  [205, 81, 214, 145],
    "P7":  [503, 357, 359, 326],
    "P8":  [367, 356, 198, 269],
    "P9":  [638, 496, 582, 466],
    "P10": [308, 357, 503, 359],
    "P11": [359, 367, 397, 351],
    "P12": [202, 370, 398, 81],
}

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import contextlib
import csv
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


# =============================================================================
# ARGUMENTOS
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bench STFP-LLM para avaliar equipes recomendadas por LLM com o surrogate."
    )
    parser.add_argument(
        "--target-projects",
        type=str,
        default=CAMINHO_PROJETOS,
        help="Caminho para target_projects.json. Padrão: target_projects.json na pasta atual.",
    )
    parser.add_argument(
        "--llm-teams",
        type=str,
        default=None,
        help="JSON/CSV externo com equipes do LLM. Se omitido, usa as equipes embutidas no script.",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Raiz do projeto STFP contendo Algorithms/, Pipeline/ e Feature_Extraction/.",
    )
    parser.add_argument(
        "--output-runs",
        type=str,
        default=CAMINHO_SAIDA_RUNS,
        help="CSV de saída com uma linha por projeto.",
    )
    parser.add_argument(
        "--output-summary",
        type=str,
        default=CAMINHO_SAIDA_RESUMO,
        help="CSV de saída com resumo por projeto.",
    )
    parser.add_argument(
        "--projects",
        type=str,
        default=",".join(PROJETOS_IDS),
        help="Lista de projetos separados por vírgula. Padrão: P1,...,P12.",
    )
    parser.add_argument(
        "--method-name",
        type=str,
        default=METODO_DEFAULT,
        help="Nome do método a registrar no CSV. Padrão: LLM_CHAT_MANUAL.",
    )
    parser.add_argument(
        "--llm-total-time-s",
        type=float,
        default=LLM_TOTAL_TIME_S_DEFAULT,
        help="Tempo total informado pelo chat, em segundos. Padrão: 407s = 6m47s.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Apaga os CSVs de saída antes de iniciar.",
    )
    return parser.parse_args()


# =============================================================================
# RESOLUÇÃO DE CAMINHOS E IMPORTAÇÃO DO PROJETO
# =============================================================================

def detectar_project_root(script_dir: Path, cli_project_root: str | None) -> Path:
    if cli_project_root:
        root = Path(cli_project_root).resolve()
        if not (root / "Pipeline").exists() or not (root / "Feature_Extraction").exists():
            raise RuntimeError(
                f"project-root informado não contém Pipeline/ e Feature_Extraction/: {root}"
            )
        return root

    candidatos = [
        Path.cwd(),
        script_dir,
        script_dir.parent,
        script_dir.parent.parent,
        script_dir.parent.parent.parent,
    ]

    for c in candidatos:
        if (c / "Pipeline").exists() and (c / "Feature_Extraction").exists():
            return c.resolve()

    raise RuntimeError(
        "Não consegui detectar a raiz do projeto STFP. "
        "Use --project-root CAMINHO_DA_PASTA_STFP."
    )


def importar_avaliador(project_root: Path):
    """Importa o avaliador surrogate e desativa logs verbosos de dimension_scoring."""
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        import Feature_Extraction.Dimension_Scoring.dimension_scoring as _ds
        _ds.DEBUG_DIM = False
    except Exception:
        pass

    from Pipeline.evaluate_teams_sur import avaliar_equipe_surrogate
    return avaliar_equipe_surrogate


@contextlib.contextmanager
def silent_stdout():
    original_stdout = sys.stdout
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            sys.stdout = devnull
            yield
    finally:
        sys.stdout = original_stdout


# =============================================================================
# LEITURA DOS PROJETOS E EQUIPES
# =============================================================================

def carregar_projetos(path: Path) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and isinstance(data.get("projects"), list):
        return {str(p["id"]): p for p in data["projects"]}

    if isinstance(data, dict):
        projetos = {}
        for pid, p in data.items():
            if isinstance(p, dict):
                p2 = dict(p)
                p2.setdefault("id", pid)
                projetos[str(pid)] = p2
        if projetos:
            return projetos

    raise ValueError(
        "Formato de target_projects.json não reconhecido. "
        "Use {'projects': [...]} ou {'P1': {...}, ...}."
    )


def project_payload(project: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "dominio": project.get("dominio", {}),
        "ecossistema": project.get("ecossistema", {}),
        "linguagens": project.get("linguagens", {}),
    }


def _as_dev_id(x: Any) -> int:
    if isinstance(x, int):
        return x
    m = re.search(r"(\d+)$", str(x))
    if not m:
        raise ValueError(f"ID de desenvolvedor inválido: {x}")
    return int(m.group(1))


def carregar_equipes_llm(path: Path | None) -> Dict[str, List[int]]:
    if path is None:
        return {pid: list(team) for pid, team in EQUIPES_LLM_DEFAULT.items()}

    if not path.exists():
        raise FileNotFoundError(f"Arquivo de equipes do LLM não encontrado: {path}")

    if path.suffix.lower() == ".csv":
        equipes: Dict[str, List[int]] = {}
        with open(path, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pid = str(row.get("project_id") or row.get("projeto") or "").strip()
                raw_team = row.get("team_ids") or row.get("equipe") or row.get("team")
                if not pid or not raw_team:
                    continue
                try:
                    parsed = json.loads(raw_team)
                except Exception:
                    parsed = [x.strip() for x in str(raw_team).split(",") if x.strip()]
                equipes[pid] = [_as_dev_id(x) for x in parsed]
        return equipes

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Formato A: [{"project_id":"P1", "team_ids":[...]}]
    if isinstance(data, list):
        equipes = {}
        for item in data:
            pid = str(item.get("project_id") or item.get("projeto") or "").strip()
            team = item.get("team_ids") or item.get("equipe") or item.get("team")
            if pid and team:
                equipes[pid] = [_as_dev_id(x) for x in team]
        return equipes

    # Formato B: {"teams":[...]}
    if isinstance(data, dict) and isinstance(data.get("teams"), list):
        equipes = {}
        for item in data["teams"]:
            pid = str(item.get("project_id") or item.get("projeto") or "").strip()
            team = item.get("team_ids") or item.get("equipe") or item.get("team")
            if pid and team:
                equipes[pid] = [_as_dev_id(x) for x in team]
        return equipes

    # Formato C: saída bruta do LLM: {"recommendations":[{"recommended_team":[{"developer_id":"Dev_1"}]}]}
    if isinstance(data, dict) and isinstance(data.get("recommendations"), list):
        equipes = {}
        for rec in data["recommendations"]:
            pid = str(rec.get("project_id") or "").strip()
            raw_team = rec.get("recommended_team") or []
            team = []
            for dev in raw_team:
                if isinstance(dev, dict):
                    team.append(_as_dev_id(dev.get("developer_id")))
                else:
                    team.append(_as_dev_id(dev))
            if pid and team:
                equipes[pid] = team
        return equipes

    # Formato D: {"P1":[...], "P2":[...]}
    if isinstance(data, dict):
        equipes = {}
        for pid, team in data.items():
            if isinstance(team, list):
                equipes[str(pid)] = [_as_dev_id(x) for x in team]
        if equipes:
            return equipes

    raise ValueError("Formato de arquivo de equipes LLM não reconhecido.")


# =============================================================================
# MÉTRICAS AUXILIARES
# =============================================================================

RUN_FIELDS = [
    # Campos compatíveis com o bench do GA/ILP
    "projeto",
    "modo",
    "status",
    "AT",
    "AC",
    "AE",
    "equipe",
    "tempo_s",
    "must_dom_ok",
    "must_eco_ok",
    "must_ling_ok",

    # Campos específicos do LLM
    "project_name",
    "metodo",
    "team_size_expected",
    "team_size_returned",
    "team_size_ok",
    "fitness_final",
    "AE_raw_surrogate",
    "all_must_ok",
    "dom_score",
    "eco_score",
    "ling_score",
    "pc_score",
    "pc_label",
    "pares_total",
    "pares_colab",
    "pares_proj",

    # Cobertura por prioridade
    "dom_M_covered", "dom_M_total",
    "eco_M_covered", "eco_M_total",
    "ling_M_covered", "ling_M_total",
    "dom_S_covered", "dom_S_total",
    "eco_S_covered", "eco_S_total",
    "ling_S_covered", "ling_S_total",
    "dom_C_covered", "dom_C_total",
    "eco_C_covered", "eco_C_total",
    "ling_C_covered", "ling_C_total",

    # Tempo
    "tempo_eval_surrogate_s",
    "tempo_llm_total_s",
    "tempo_llm_por_projeto_s",
    "tempo_llm_is_approx",

    "timestamp",
]

SUMMARY_FIELDS = [
    "projeto", "project_name", "metodo", "modo",
    "status", "team_size_expected", "team_size_returned", "team_size_ok",
    "AE", "AT", "AC", "fitness_final", "AE_raw_surrogate",
    "all_must_ok", "must_dom_ok", "must_eco_ok", "must_ling_ok",
    "dom_score", "eco_score", "ling_score", "pc_score", "pc_label",
    "pares_total", "pares_colab", "pares_proj",
    "equipe",
    "tempo_s", "tempo_eval_surrogate_s", "tempo_llm_total_s",
    "tempo_llm_por_projeto_s", "tempo_llm_is_approx",
]


def coverage_count(eval_res: Dict[str, Any], dim: str, priority: str) -> Tuple[Any, Any]:
    cov = eval_res.get("coverage", {})
    if not isinstance(cov, dict):
        return "", ""
    d = cov.get(dim, {})
    if not isinstance(d, dict):
        return "", ""
    p = d.get(priority)
    if p is None:
        return "", ""
    if isinstance(p, dict):
        return p.get("covered", ""), p.get("total", "")
    return "", ""


def must_flags_from_eval(eval_res: Dict[str, Any]) -> Dict[str, bool]:
    flags = {
        "must_dom_ok": True,
        "must_eco_ok": True,
        "must_ling_ok": True,
    }
    dim_to_flag = {
        "dominio": "must_dom_ok",
        "ecossistema": "must_eco_ok",
        "linguagens": "must_ling_ok",
    }

    coverage = eval_res.get("coverage", {})
    if isinstance(coverage, dict):
        for dim, flag_name in dim_to_flag.items():
            m = coverage.get(dim, {}).get("M")
            if isinstance(m, dict):
                total = int(m.get("total", 0) or 0)
                covered = int(m.get("covered", 0) or 0)
                if total > 0 and covered < total:
                    flags[flag_name] = False
        flags["all_must_ok"] = all(flags.values())
        return flags

    scores = eval_res.get("scores", {})
    if isinstance(scores, dict):
        for dim, flag_name in dim_to_flag.items():
            dbg = scores.get(dim, {}).get("debug", {})
            if not isinstance(dbg, dict):
                continue
            feats = dbg.get("featsM") or dbg.get("feats")
            if not isinstance(feats, dict):
                continue
            covp = float(feats.get("covP", 1.0) or 0.0)
            if covp < 1.0:
                flags[flag_name] = False

    flags["all_must_ok"] = all(flags.values())
    return flags


def _score(eval_res: Dict[str, Any], dim: str) -> Any:
    scores = eval_res.get("scores", {})
    if isinstance(scores, dict):
        d = scores.get(dim, {})
        if isinstance(d, dict):
            return d.get("score", "")
    return ""


def ensure_csv(path: Path, fields: List[str], restart: bool = False) -> None:
    if restart and path.exists():
        path.unlink()
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()


def write_csv(path: Path, fields: List[str], rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def to_float(x: Any) -> float | None:
    try:
        if x in (None, ""):
            return None
        return float(x)
    except Exception:
        return None


def fmt_num(x: Any, ndigits: int = 5) -> str:
    v = to_float(x)
    return f"{v:.{ndigits}f}" if v is not None else "-"


# =============================================================================
# EXECUÇÃO DO BENCH
# =============================================================================

def avaliar_time_llm(
    avaliar_equipe_surrogate,
    projeto: Dict[str, Any],
    team: List[int],
    metodo: str,
    tempo_llm_por_projeto_s: float,
    tempo_llm_total_s: float,
) -> Dict[str, Any]:
    team_size_expected = int(projeto.get("team_size", 4))
    team_size_returned = len(team)
    team_size_ok = team_size_returned == team_size_expected

    t0 = time.perf_counter()
    with silent_stdout():
        eval_res = avaliar_equipe_surrogate(team, project_payload(projeto), log=False)
    tempo_eval = time.perf_counter() - t0

    flags = must_flags_from_eval(eval_res)
    ae_raw = float(eval_res.get("media_AE", 0.0))
    fitness_final = ae_raw if flags["all_must_ok"] else HARD_MUST_PENALTY

    pares_info = eval_res.get("pares_info", {}) if isinstance(eval_res.get("pares_info"), dict) else {}

    row: Dict[str, Any] = {
        "projeto": projeto.get("id", ""),
        "modo": MODO,
        "status": "OK",
        "AT": eval_res.get("AT_cont", ""),
        "AC": eval_res.get("AC_cont", ""),
        "AE": ae_raw,
        "equipe": json.dumps(team, ensure_ascii=False),
        "tempo_s": tempo_llm_por_projeto_s,
        "must_dom_ok": flags["must_dom_ok"],
        "must_eco_ok": flags["must_eco_ok"],
        "must_ling_ok": flags["must_ling_ok"],
        "project_name": projeto.get("name", ""),
        "metodo": metodo,
        "team_size_expected": team_size_expected,
        "team_size_returned": team_size_returned,
        "team_size_ok": team_size_ok,
        "fitness_final": fitness_final,
        "AE_raw_surrogate": ae_raw,
        "all_must_ok": flags["all_must_ok"],
        "dom_score": _score(eval_res, "dominio"),
        "eco_score": _score(eval_res, "ecossistema"),
        "ling_score": _score(eval_res, "linguagens"),
        "pc_score": eval_res.get("pc_score", ""),
        "pc_label": eval_res.get("pc_label", ""),
        "pares_total": pares_info.get("n_total", ""),
        "pares_colab": pares_info.get("n_colab", ""),
        "pares_proj": pares_info.get("n_proj", ""),
        "tempo_eval_surrogate_s": round(tempo_eval, 6),
        "tempo_llm_total_s": tempo_llm_total_s,
        "tempo_llm_por_projeto_s": tempo_llm_por_projeto_s,
        "tempo_llm_is_approx": True,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    for dim_key, prefix in [
        ("dominio", "dom"),
        ("ecossistema", "eco"),
        ("linguagens", "ling"),
    ]:
        for prio in ["M", "S", "C"]:
            covered, total = coverage_count(eval_res, dim_key, prio)
            row[f"{prefix}_{prio}_covered"] = covered
            row[f"{prefix}_{prio}_total"] = total

    return row


def gerar_resumo(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    resumo = []
    for r in rows:
        resumo.append({k: r.get(k, "") for k in SUMMARY_FIELDS})
    return resumo


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    target_projects_path = Path(args.target_projects).resolve()
    if not target_projects_path.exists():
        raise FileNotFoundError(f"Arquivo de projetos não encontrado: {target_projects_path}")

    llm_teams_path = Path(args.llm_teams).resolve() if args.llm_teams else None
    output_runs = Path(args.output_runs).resolve()
    output_summary = Path(args.output_summary).resolve()

    if args.restart:
        for p in [output_runs, output_summary]:
            if p.exists():
                p.unlink()

    project_root = detectar_project_root(script_dir, args.project_root)
    projetos = carregar_projetos(target_projects_path)
    equipes_llm = carregar_equipes_llm(llm_teams_path)

    projetos_ids = [p.strip() for p in args.projects.split(",") if p.strip()]
    ids_invalidos = [pid for pid in projetos_ids if pid not in projetos]
    if ids_invalidos:
        print(f"[BENCH-LLM] AVISO: projetos não encontrados em target_projects: {ids_invalidos}")
    ids_sem_equipe = [pid for pid in projetos_ids if pid not in equipes_llm]
    if ids_sem_equipe:
        print(f"[BENCH-LLM] AVISO: projetos sem equipe LLM: {ids_sem_equipe}")

    ids_validos = [pid for pid in projetos_ids if pid in projetos and pid in equipes_llm]
    if not ids_validos:
        raise RuntimeError("Nenhum projeto válido para avaliar.")

    tempo_llm_total_s = float(args.llm_total_time_s or 0.0)
    tempo_llm_por_projeto_s = tempo_llm_total_s / len(ids_validos) if tempo_llm_total_s > 0 else ""

    print(f"\n{'#'*70}")
    print(f"#  BENCH STFP-LLM  |  Modo: {MODO}")
    print(f"#  Projetos: {ids_validos}")
    print(f"#  Método: {args.method_name}")
    print(f"#  Tempo LLM total informado: {tempo_llm_total_s:.2f}s")
    if tempo_llm_por_projeto_s != "":
        print(f"#  Tempo LLM por projeto: {tempo_llm_por_projeto_s:.4f}s (aprox.)")
    print(f"{'#'*70}\n")

    print(f"[BENCH-LLM] project_root    : {project_root}")
    print(f"[BENCH-LLM] target_projects : {target_projects_path}")
    print(f"[BENCH-LLM] output_runs     : {output_runs}")
    print(f"[BENCH-LLM] output_summary  : {output_summary}\n")

    avaliar_equipe_surrogate = importar_avaliador(project_root)

    rows: List[Dict[str, Any]] = []
    for i, pid in enumerate(ids_validos, 1):
        projeto = dict(projetos[pid])
        projeto.setdefault("id", pid)
        team = equipes_llm[pid]

        try:
            row = avaliar_time_llm(
                avaliar_equipe_surrogate=avaliar_equipe_surrogate,
                projeto=projeto,
                team=team,
                metodo=args.method_name,
                tempo_llm_por_projeto_s=float(tempo_llm_por_projeto_s) if tempo_llm_por_projeto_s != "" else "",
                tempo_llm_total_s=tempo_llm_total_s,
            )
        except Exception as e:
            row = {
                "projeto": pid,
                "modo": MODO,
                "status": f"Erro: {e}",
                "AT": "", "AC": "", "AE": "",
                "equipe": json.dumps(team, ensure_ascii=False),
                "tempo_s": tempo_llm_por_projeto_s,
                "must_dom_ok": "", "must_eco_ok": "", "must_ling_ok": "",
                "project_name": projeto.get("name", ""),
                "metodo": args.method_name,
                "team_size_expected": projeto.get("team_size", 4),
                "team_size_returned": len(team),
                "team_size_ok": len(team) == int(projeto.get("team_size", 4)),
                "fitness_final": "",
                "AE_raw_surrogate": "",
                "all_must_ok": "",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }

        rows.append(row)

        print(
            f"  [{i:02d}/{len(ids_validos):02d}] {pid} "
            f"AE={fmt_num(row.get('AE'))} AT={fmt_num(row.get('AT'))} AC={fmt_num(row.get('AC'))} "
            f"must_ok={row.get('all_must_ok')} equipe={row.get('equipe')}"
        )

    write_csv(output_runs, RUN_FIELDS, rows)
    resumo = gerar_resumo(rows)
    write_csv(output_summary, SUMMARY_FIELDS, resumo)

    validos = [r for r in rows if str(r.get("all_must_ok", "")).lower() == "true"]
    aes = [to_float(r.get("AE")) for r in rows if to_float(r.get("AE")) is not None]
    ats = [to_float(r.get("AT")) for r in rows if to_float(r.get("AT")) is not None]
    acs = [to_float(r.get("AC")) for r in rows if to_float(r.get("AC")) is not None]

    print(f"\n\n{'='*70}")
    print("  RESUMO FINAL — LLM")
    print(f"{'='*70}")
    print(f"Projetos avaliados       : {len(rows)}")
    print(f"Projetos hard-must OK    : {len(validos)}")
    print(f"Taxa hard-must OK        : {len(validos) / len(rows):.4f}")
    print(f"AE médio                 : {statistics.mean(aes):.5f}" if aes else "AE médio                 : -")
    print(f"AT médio                 : {statistics.mean(ats):.5f}" if ats else "AT médio                 : -")
    print(f"AC médio                 : {statistics.mean(acs):.5f}" if acs else "AC médio                 : -")
    print(f"Tempo LLM total informado: {tempo_llm_total_s:.2f}s")

    print(f"\n[BENCH-LLM] CSV de execuções salvo em: {output_runs}")
    print(f"[BENCH-LLM] CSV de resumo salvo em    : {output_summary}")


if __name__ == "__main__":
    main()
