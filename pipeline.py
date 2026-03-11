#!/usr/bin/env python3
"""
Pipeline: Gap Detection in Customer Insights
1. Filter insights (count >= 500)
2. LLM scoring (pain_score, clarity_score, problematique)
3. Priority sort: pain_score × (11 - clarity_score)
4. Survey generation (destinataire, questions, échantillon)
5. Output: brief.json + brief.md
"""

import json
import os
import time
from dotenv import load_dotenv
import anthropic

load_dotenv()

# --- Config ---
DATA_PATH = "data/data.json"
JOURNEY_PATH = "data/journey.json"
ORGCHART_PATH = "data/orgchart.json"
OUTPUT_JSON = "output/brief.json"
OUTPUT_MD = "output/brief.md"
MIN_COUNT = 500
BATCH_SIZE = 10
MODEL = "claude-haiku-4-5"

# --- System prompt for Step 2: Scoring ---
SCORING_SYSTEM_PROMPT = """Tu es un analyste expert en expérience client. Tu reçois des insights
extraits de retours clients d'un lieu de divertissement (casino).

Pour chaque insight, tu dois produire exactement 3 éléments:

1. **pain_score** (1-10): La gravité du problème pour le client.
   - Prends en compte le volume de mentions ET l'intensité émotionnelle.
   - 1 = gêne mineure, peu de monde concerné
   - 5 = frustration notable, impact modéré sur l'expérience
   - 10 = rupture totale, le client ne revient pas

2. **clarity_score** (1-10): À quel point la source du problème est comprise.
   - Analyse les sous-insights (children): est-ce qu'ils pointent tous
     vers la même cause, ou est-ce qu'ils partent dans des directions
     différentes?
   - 1 = on ne comprend pas du tout l'origine du problème, les
     sous-insights sont vagues ou contradictoires
   - 5 = on a une idée générale mais les causes précises restent floues
   - 10 = la cause racine est parfaitement identifiée et actionnable

3. **problematique** (1-2 phrases): La question qu'il faudrait poser AUX CLIENTS
   pour mieux comprendre leur insatisfaction ou leur perception.
   Ce n'est PAS une question opérationnelle interne. C'est la question
   qui cible ce qu'on ne comprend pas encore dans leur perception,
   leurs attentes, leurs critères de jugement, ou l'impact sur leur comportement.

Règles:
- Sois factuel. Base-toi uniquement sur les données fournies.
- Un pain_score élevé + clarity_score bas = priorité maximale pour la marque.
- La problématique ne doit PAS reformuler ce qu'on sait déjà. Elle doit pointer
  vers ce qu'on NE SAIT PAS côté client.
- Réponds en JSON strict, sans markdown, sans backticks."""

# --- System prompt for Step 4: Survey generation ---
SURVEY_SYSTEM_PROMPT = """Tu es un expert en research design et en organisation d'entreprise.
Tu reçois des insights clients scorés provenant d'un lieu de divertissement (casino),
ainsi que l'organigramme de l'entreprise.

Pour chaque insight, tu dois produire exactement 3 éléments:

1. **destinataire_interne**: Le rôle ou poste de la personne dans l'organigramme fourni
   qui doit recevoir les résultats de cette investigation et qui a le pouvoir d'agir.
   Tu DOIS choisir parmi les personnes listées dans l'organigramme. Indique le nom et le rôle
   exact, et justifie pourquoi cette personne est la bonne.

2. **questions_survey**: Une liste de 3 à 5 questions ouvertes à poser aux clients par
   un agent vocal IA. Les questions doivent être:
   - Conversationnelles (comme si un humain parlait au téléphone, pas un questionnaire formel)
   - Progresser du général au spécifique
   - Couvrir à la fois le pourquoi de l'insatisfaction et ce qui rendrait le client satisfait
   - Directement liées à la problématique identifiée

3. **taille_echantillon**: Le nombre de clients à contacter pour obtenir des résultats
   exploitables. Justifie le chiffre en fonction de la complexité du sujet et du
   clarity_score (un clarity_score bas nécessite plus de répondants car le problème est
   moins compris).

Règles:
- Les questions doivent être formulées comme si un humain parlait au téléphone.
- Chaque question doit creuser un angle différent, pas de redondance.
- Le destinataire interne doit être choisi DANS l'organigramme fourni.
- Réponds en JSON strict, sans markdown, sans backticks."""


def load_data():
    with open(DATA_PATH, "r") as f:
        return json.load(f)


def load_orgchart():
    with open(ORGCHART_PATH, "r") as f:
        return json.load(f)


def filter_insights(data):
    """Keep only insights with count >= MIN_COUNT at parent level."""
    return [d for d in data if d["count"] >= MIN_COUNT]


def format_insight_for_llm(insight):
    """Format a single insight for the LLM prompt."""
    children_text = ""
    if insight.get("children"):
        children_lines = []
        for i, child in enumerate(insight["children"], 1):
            children_lines.append(
                f"  {i}. (count: {child['count']}) {child['long_description']}"
            )
        children_text = "\n".join(children_lines)

    return f"""--- Insight ---
Type: {insight['type']}
Groupe (étape parcours): {insight['group']}
Tag (touchpoint): {insight['tag']}
Count (mentions): {insight['count']}
Description: {insight['longDescription']}
Sous-insights:
{children_text}
"""


def build_scoring_prompt(batch):
    """Build the user prompt for scoring a batch of insights."""
    insights_text = "\n".join(format_insight_for_llm(i) for i in batch)

    return f"""Analyse les insights suivants et produis pour chacun un pain_score, clarity_score et problematique.

Réponds en JSON strict (array d'objets):
[
  {{
    "tag": "<nom du touchpoint>",
    "type": "<type de l'insight>",
    "pain_score": <1-10>,
    "clarity_score": <1-10>,
    "problematique": "<question orientée client à creuser>"
  }}
]

---
Insights à analyser:

{insights_text}"""


def build_survey_prompt(enriched_insights, orgchart):
    """Build the user prompt for survey generation."""
    orgchart_text = json.dumps(orgchart, ensure_ascii=False, indent=2)

    insights_text = ""
    for item in enriched_insights:
        insights_text += f"""--- Insight scoré ---
Tag: {item['tag']}
Type: {item['type']}
Groupe: {item['group']}
Mentions: {item['count']}
Pain score: {item['pain_score']}/10
Clarity score: {item['clarity_score']}/10
Priorité: {item['priority']}
Problématique: {item['problematique']}
Description: {item['longDescription']}

"""

    return f"""Voici l'organigramme de l'entreprise:

{orgchart_text}

---

À partir des insights scorés suivants, génère pour chacun le destinataire interne
(choisi DANS l'organigramme ci-dessus), les questions du survey vocal et la taille d'échantillon.

Réponds en JSON strict (array d'objets):
[
  {{
    "tag": "<nom du touchpoint>",
    "destinataire_interne": {{
      "role": "<titre exact de l'organigramme>",
      "nom": "<nom de la personne>",
      "justification": "<pourquoi cette personne>"
    }},
    "questions_survey": [
      "<question 1>",
      "<question 2>",
      "<question 3>"
    ],
    "taille_echantillon": {{
      "nombre": <nombre>,
      "justification": "<pourquoi ce nombre>"
    }}
  }}
]

---
Insights scorés à traiter:

{insights_text}"""


def parse_llm_json(raw):
    """Parse JSON from LLM response, handling potential markdown wrapping."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start_idx = raw.find("[")
        end_idx = raw.rfind("]") + 1
        if start_idx != -1 and end_idx > start_idx:
            return json.loads(raw[start_idx:end_idx])
        raise


def score_batch(client, batch, batch_num, total_batches):
    """Send a batch to the LLM for scoring and parse the response."""
    prompt = build_scoring_prompt(batch)

    print(f"  Batch {batch_num}/{total_batches} ({len(batch)} insights)...", end=" ", flush=True)
    start = time.time()

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=SCORING_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text
    elapsed = time.time() - start
    print(f"done ({elapsed:.1f}s, {response.usage.input_tokens}in/{response.usage.output_tokens}out)")

    try:
        scores = parse_llm_json(raw)
    except (json.JSONDecodeError, ValueError):
        print(f"    WARNING: Could not parse JSON from batch {batch_num}")
        print(f"    Raw response: {raw[:200]}")
        scores = []

    return scores


def generate_surveys(client, enriched_insights, orgchart):
    """Send scored insights to LLM for survey generation."""
    prompt = build_survey_prompt(enriched_insights, orgchart)

    print(f"  Generating surveys for {len(enriched_insights)} insights...", end=" ", flush=True)
    start = time.time()

    response = client.messages.create(
        model=MODEL,
        max_tokens=8192,
        system=SURVEY_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text
    elapsed = time.time() - start
    print(f"done ({elapsed:.1f}s, {response.usage.input_tokens}in/{response.usage.output_tokens}out)")

    try:
        surveys = parse_llm_json(raw)
    except (json.JSONDecodeError, ValueError):
        print(f"    WARNING: Could not parse survey JSON")
        print(f"    Raw response: {raw[:300]}")
        surveys = []

    return surveys


def compute_priority(scored_insights):
    """Compute priority = pain_score × (11 - clarity_score) and sort."""
    for item in scored_insights:
        item["priority"] = item["pain_score"] * (11 - item["clarity_score"])
    return sorted(scored_insights, key=lambda x: x["priority"], reverse=True)


def enrich_with_original_data(scored_insights, filtered_data):
    """Add original data fields back to scored insights."""
    lookup = {}
    for d in filtered_data:
        key = (d["tag"], d["type"])
        lookup[key] = d

    enriched = []
    for item in scored_insights:
        key = (item["tag"], item["type"])
        original = lookup.get(key, {})
        enriched.append({
            "tag": item["tag"],
            "type": item["type"],
            "group": original.get("group", ""),
            "count": original.get("count", 0),
            "longDescription": original.get("longDescription", ""),
            "pain_score": item["pain_score"],
            "clarity_score": item["clarity_score"],
            "priority": item["priority"],
            "problematique": item["problematique"],
            "children_count": len(original.get("children", [])),
        })
    return enriched


def merge_survey_data(enriched, surveys):
    """Merge survey generation results into enriched insights."""
    survey_lookup = {s["tag"]: s for s in surveys}

    for item in enriched:
        survey = survey_lookup.get(item["tag"], {})
        item["destinataire_interne"] = survey.get("destinataire_interne", {})
        item["questions_survey"] = survey.get("questions_survey", [])
        item["taille_echantillon"] = survey.get("taille_echantillon", {})

    return enriched


def generate_markdown(enriched):
    """Generate a readable brief.md from enriched data."""
    lines = [
        "# Brief de Recherche Client — Gaps Prioritaires",
        "",
        f"> Généré automatiquement | {len(enriched)} insights scorés | Tri par priorité décroissante",
        "",
        "## Matrice Pain × Clarity",
        "",
        "| # | Tag | Type | Pain | Clarity | Priorité | Destinataire |",
        "|---|-----|------|------|---------|----------|--------------|",
    ]

    for i, item in enumerate(enriched, 1):
        dest = item.get("destinataire_interne", {})
        dest_name = dest.get("nom", "—")
        lines.append(
            f"| {i} | {item['tag']} | {item['type'][:20]} | "
            f"{item['pain_score']}/10 | {item['clarity_score']}/10 | "
            f"**{item['priority']}** | {dest_name} |"
        )

    lines.extend(["", "---", "", "## Détail des Problématiques & Surveys", ""])

    for i, item in enumerate(enriched, 1):
        pain_bar = "🔴" * item["pain_score"] + "⚪" * (10 - item["pain_score"])
        clarity_bar = "🟢" * item["clarity_score"] + "⚪" * (10 - item["clarity_score"])

        dest = item.get("destinataire_interne", {})
        echantillon = item.get("taille_echantillon", {})

        lines.extend([
            f"### {i}. {item['tag']} — {item['type']}",
            f"**Groupe:** {item['group']} | **Mentions:** {item['count']} | **Sous-clusters:** {item['children_count']}",
            "",
            f"- Pain: {pain_bar} ({item['pain_score']}/10)",
            f"- Clarity: {clarity_bar} ({item['clarity_score']}/10)",
            f"- **Priorité: {item['priority']}**",
            "",
            f"> **Problématique:** {item['problematique']}",
            "",
            f"*Description:* {item['longDescription']}",
            "",
        ])

        # Destinataire
        if dest:
            lines.extend([
                f"**Destinataire:** {dest.get('nom', '—')} — {dest.get('role', '—')}",
                f"*Justification:* {dest.get('justification', '—')}",
                "",
            ])

        # Survey questions
        questions = item.get("questions_survey", [])
        if questions:
            lines.append("**Questions pour l'agent vocal:**")
            for q_idx, q in enumerate(questions, 1):
                lines.append(f"{q_idx}. {q}")
            lines.append("")

        # Échantillon
        if echantillon:
            lines.extend([
                f"**Échantillon recommandé:** {echantillon.get('nombre', '—')} clients",
                f"*{echantillon.get('justification', '')}*",
                "",
            ])

        lines.extend(["---", ""])

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("PIPELINE: Gap Detection + Survey Generation")
    print("=" * 60)

    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("\nERROR: ANTHROPIC_API_KEY not set in .env")
        print("Add your key to .env: ANTHROPIC_API_KEY=sk-ant-...")
        return

    client = anthropic.Anthropic(api_key=api_key)

    # Step 1: Load & filter
    print("\n[1/5] Loading and filtering data...")
    data = load_data()
    filtered = filter_insights(data)
    orgchart = load_orgchart()
    print(f"  {len(data)} total → {len(filtered)} with count >= {MIN_COUNT}")
    print(f"  Organigramme: {len(orgchart['organigramme'])} postes chargés")

    types = {}
    for d in filtered:
        types[d["type"]] = types.get(d["type"], 0) + 1
    for t, c in sorted(types.items(), key=lambda x: -x[1]):
        print(f"    {t}: {c}")

    # Step 2: LLM scoring
    print(f"\n[2/5] Scoring with {MODEL} (batches of {BATCH_SIZE})...")
    batches = [filtered[i:i + BATCH_SIZE] for i in range(0, len(filtered), BATCH_SIZE)]
    all_scores = []
    total_start = time.time()

    for i, batch in enumerate(batches, 1):
        scores = score_batch(client, batch, i, len(batches))
        all_scores.extend(scores)

    scoring_elapsed = time.time() - total_start
    print(f"  Total: {len(all_scores)} scored in {scoring_elapsed:.1f}s")

    # Step 3: Priority sort
    print("\n[3/5] Computing priorities...")
    sorted_scores = compute_priority(all_scores)
    enriched = enrich_with_original_data(sorted_scores, filtered)
    print(f"  Top 5 priorities:")
    for item in enriched[:5]:
        print(f"    [{item['priority']}] {item['tag']} ({item['type'][:25]}) — pain:{item['pain_score']} clarity:{item['clarity_score']}")

    # Step 4: Survey generation
    print(f"\n[4/5] Generating surveys with {MODEL}...")
    survey_start = time.time()
    surveys = generate_surveys(client, enriched, orgchart)
    survey_elapsed = time.time() - survey_start
    print(f"  {len(surveys)} survey briefs generated in {survey_elapsed:.1f}s")

    # Merge survey data
    enriched = merge_survey_data(enriched, surveys)

    # Step 5: Output
    print("\n[5/5] Writing outputs...")
    os.makedirs("output", exist_ok=True)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)
    print(f"  {OUTPUT_JSON} ({len(enriched)} entries)")

    md_content = generate_markdown(enriched)
    with open(OUTPUT_MD, "w") as f:
        f.write(md_content)
    print(f"  {OUTPUT_MD}")

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print("DONE")
    print(f"  Insights scored: {len(enriched)}")
    if enriched:
        print(f"  Priority range: {enriched[-1]['priority']} - {enriched[0]['priority']}")
        avg_pain = sum(i["pain_score"] for i in enriched) / len(enriched)
        avg_clarity = sum(i["clarity_score"] for i in enriched) / len(enriched)
        print(f"  Avg pain: {avg_pain:.1f} | Avg clarity: {avg_clarity:.1f}")
        print(f"  #1 priority: {enriched[0]['tag']} → {enriched[0].get('destinataire_interne', {}).get('nom', '?')}")
    print(f"  Scoring: {scoring_elapsed:.1f}s | Surveys: {survey_elapsed:.1f}s | Total: {total_elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
