# Language Factory author form — example

The author form is the only thing a human writes before `tau2 factory draft`
generates a complete `pack_draft.yaml` + voice guidelines. It is a YAML
mapping, either in a plain `.yaml` file or (as here) inside the ```yaml
fenced block of a markdown file — `tau2 factory draft --lang <lang> --form
<this file>` parses the first ```yaml block it finds.

Fields (validated by `tau2.multilingual.factory.author_form.AuthorForm` —
the field list below cannot drift from the code, run
`python -c "from tau2.multilingual.factory.author_form import AuthorForm, render_field_docs; print(render_field_docs(AuthorForm))"`
for the live version):

- `language` (required): ISO 639-1 code, e.g. `sw`. Must match `--lang`.
- `display_name` (required): human-readable language name.
- `script` (required): ISO 15924 script code (`latn`, `deva`, `arab`, ...).
  Must have a registered Unicode range in
  `tau2.multilingual.invariants.SCRIPT_RANGES` — extending that for a new
  script is an engineer task, not a factory task.
- `locale_notes` (optional): freeform notes on dialect/locale conventions
  the drafted personas and guidelines should respect.
- `personas` (required): 2–4 sketches, each with a `name` and a freeform
  `sketch`. Guidance from the Hindi build: span a register axis
  (high vs. low code-switching, different sociolects) and any major accent
  divide — that buys the most realism per persona.
- `preferences` (optional): freeform drafting preferences (backchannel
  density, which persona is mobile/out-and-about vs. at-home — that only steers
  the persona's `acoustic_preset_id`, since the acoustic environment is implied
  by the chosen background bed, not tagged separately — anything else the
  drafter should honor).

```yaml
language: sw
display_name: Swahili
script: latn
locale_notes: >-
  Coastal Tanzanian Swahili as the baseline; one persona should lean
  Nairobi-style Sheng-adjacent code-switching. Phone numbers are read in
  pairs; English letter names are used when spelling booking codes.
personas:
  - name: Amara
    sketch: >-
      Late 20s, Dar es Salaam, marketing professional. Fast, confident
      Swahili with heavy English insertions for work/tech/money terms.
      High tolerance for an English-only agent — switches into fluent
      English without complaint. Usually calls from traffic or a busy
      office. Brisk closings.
  - name: Baraka
    sketch: >-
      Early 60s, retired teacher near Mombasa. Slow, courteous, almost
      pure Swahili — only numbers and brand names in English. Low
      tolerance for an English-only agent: politely insists on Swahili.
      Calls from a quiet household. Long, warm, multi-turn goodbyes.
preferences: >-
  Swahili listeners backchannel actively ("eeh", "sawa", "ndiyo") —
  noticeably more often than English. Prefer the generic noise files until
  locale recordings are sourced.
```

After drafting, iterate with `tau2 factory validate --draft --lang sw` and
direct edits to `data/tau2/multilingual/_factory/sw/pack_draft.yaml`.
