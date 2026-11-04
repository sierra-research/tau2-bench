# Preserve personal-name gender during localization

A localized name must preserve the source person's gender constraints. Those
constraints belong to each source identity, not the translated spelling or the
caller voice. The factory never changes task/persona gender to accommodate a name.
Aliases are joined by source owner and role; identical English spellings belonging
to different people remain separate.

`name_gender` records source evidence and the localized name assessment. Reviewed
source metadata takes precedence. Where banking lacks a gender field, the source
assessment distinguishes explicit co-referential context from a qualified name
convention. Ambiguous and unisex source names do not acquire a fabricated binary
gender. Compatible unisex target names are allowed; uncertainty remains visible
for native review. The evidence is bound to the exact localized forms, and editing
those forms invalidates it. Independent names in a list use indexed
`name_gender_items` evidence, so a deliberate wrong-name decoy does not inherit the
correct name's gender.

Previously approved donor names cannot be reused without evidence that they fit
the new source person's gender. Missing compatible donors fall back to fixed-prompt
authoring. The existing human approval of an exact map remains necessary; a model
assessment is not native-speaker approval. Banking publication rejects changed
personal names without assessment coverage.

## Native review

Native reviewers should check inferred source-name associations, target-name
compatibility, family/given components, and retained name difficulty where relevant.
Unknown source gender is an explicit review question, not evidence of a mismatch.
Existing review exports are stale for changed rows until separately regenerated;
prior approvals must not be transferred onto revised forms.
