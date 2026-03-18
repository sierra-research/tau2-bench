NORMALIZE_FOR_WER_PROMPT = """
<Task>
You are normalizing transcripts for Word Error Rate (WER) calculation. Given two transcripts (expected and actual), normalize both to a consistent format to ensure fair comparison.
</Task>

<Normalization Rules>
1. Convert all text to lowercase
2. Expand contractions (e.g., "don't" → "do not", "it's" → "it is")
3. Convert numbers to word form in the correct language (e.g., "2" → "two", "1st" → "first", "101" -> "ciento uno")
4. Remove all punctuation except apostrophes in possessives
5. Normalize spacing (single spaces between words). For japanese and similar languages that don't have words, add spaces between characters
6. Expand common abbreviations (e.g., "Dr." → "doctor", "Mr." → "mister")
7. Normalize variant spellings to consistent forms (e.g., "okay"/"ok" → "okay")
8. Keep spelled-out letters/numbers as is, normalized between both transcripts (e.g., "B-E-E" stays as "b e e")
9. Normalize filler words consistently (e.g., "um", "uh", "hmm" → remove all)
10. Handle empty/missing transcripts appropriately
11. Use consistent alphabetization when appropriate ("日本" -> "にほん")
</Normalization Rules>

<Guidelines>
- Apply the EXACT same normalization rules to both transcripts
- Preserve the semantic content while normalizing format
- If one transcript is empty/missing, normalize the other normally and return empty string ("") for the empty one
- Be consistent in your normalization choices
- Empty transcripts should be normalized to empty string (""), not null
</Guidelines>

<Examples>
Input:
Expected: "Hello, my ID is 123-ABC."
Actual: "hello my i.d. is one two three a b c"

Output:
{{
  "normalized_expected": "hello my id is one two three a b c",
  "normalized_actual": "hello my id is one two three a b c"
}}

Input:
Expected: "Dr. Smith's office, 2nd floor."
Actual: "Doctor Smith office second floor"

Output:
{{
  "normalized_expected": "doctor smiths office second floor",
  "normalized_actual": "doctor smith office second floor"
}}

Input:
Expected: "Please enter your account number."
Actual: ""

Output:
{{
  "normalized_expected": "please enter your account number",
  "normalized_actual": ""
}}
</Examples>

<Input>
Expected:
{expected_transcript}

Actual:
{actual_transcript}
</Input>

<Output>
Return a JSON object with two fields: "normalized_expected" and "normalized_actual" containing the normalized versions of the input transcripts.
</Output>
""".strip()

SIGNIFICANT_WORD_ERRORS_PROMPT = """
<Task>
You are evaluating the quality of a transcription task. You are given two strings:
- Expected: The correct/reference transcription.
- Actual: The transcription produced by a model.

Additionally, you are given the words where the mistranscription occurred.
The mistranscription can be in the form of a substitution, deletion, or insertion.

Your job is to compare the two words and return a JSON object with a 'score' field for every error. This field must be an integer from 1 to 3, indicating how closely the actual transcription matches the expected one.
</Task>

<Scoring Criteria>
- Score 1 – Significant Error  
  The meaning of the whole sentence is derailed or incoherent because of the error.

- Score 2 – Minor Error  
  The meaning of the overall sentence doesn't change. The words may be completely different in meaning, but it doesnt derail the meaning of the whole sentence.

- Score 3 – No Error  
  The words are semantically the same.
</Scoring Criteria>

<Guidance>
- Ignore differences in:
  - Punctuation (e.g., periods, commas, dashes)
  - Capitalization (e.g., "hello" vs. "HELLO")
  - Spacing (e.g., "icecream" vs. "ice cream")
- Treat numeric equivalents as equal (e.g., "2" == "two")
- Consider equivalent variants (e.g., "z" == "zed")
- Check for correctness of inputs, such as names, email addresses, spellings, phone numbers, etc. If it is spelled out letters or numbers (H E N R Y), a single letter difference is considered a significant error. If it is a proper noun as a whole (henry), a single letter difference should be considered a minor error.
- If the actual transcription is in a different language, it is a major error, and should be scored as 1.
- For spelled out words, check for correctness of spelling, but differences in format (all caps, spaces, etc.) should be ignored.
- If a transcript is inaudible, this is not considered missing.
- If the expected transcript contains "<unintelligible>", that comparison should receive a score of 3. 
</Guidance>
</Task>

<Input>
Expected:
{expected_transcript}

Actual:
{actual_transcript}
</Input>

<Examples>
<Example>
Expected:
I want to check my balance

Actual:
I wanted to check my balance

Response:
{{
  "scores": [
    {{
      "error": "want substituted for wanted",
      "score": 3,
    }}
  ]
}}
</Example>
<Example>
Expected:
That is right

Actual:
Thats not right, no

Response:
{{
  "scores": [
    {{
      "error": "Substitution: 'that' to 'thats' at position 0",
      "score": 3,
    }},
    {{
      "error": "Substitution: 'is' to 'not' at position 2",
      "score": 1,
    }},
    {{
      "error": "Insertion: 'no' at position 4",
      "score": 1,
    }}
  ]
}}
</Example>
<Example>
Expected: 
john franklin

Actual:
john franklim

Response:
{{
  "scores": [
    {{
      "error": "Substitution: 'franklin' to 'franklim'",
      "score": 2,
    }}
  ]
}}
</Example>
<Example>
Expected: 
R O S E

Actual:
R O S D

Response:
{{
  "scores": [
    {{
      "error": "Substitution: 'D' to 'E'",
      "score": 1,
    }}
  ]
}}
</Example>

</Examples>

<Input>
Expected:
{expected_transcript}

Actual:
{actual_transcript}
</Input>

<Output>
Return a JSON object.
{{
  "scores": [
    {errors}
  ]
}}
</Output>
""".strip()

SCORE_TRANSCRIPT_PROMPT = """
<Task>
You are evaluating the quality of a transcription task. You are given two strings:
- **Expected**: The correct/reference transcription.
- **Actual**: The transcription produced by a model or annotator.

Your job is to compare the two strings and return a JSON object with a 'score' field. This field must be an integer from 0 to 3, indicating how closely the actual transcription matches the expected one.
</Task>

<Scoring Criteria>
- **Score 0** – *Missing Transcription*  
  The actual output is empty, null, or contains no recognizable speech content at all.

- **Score 1** – *Really Bad but Not Missing*  
  The actual output contains some speech content but is largely incorrect, with most words being wrong, the intent being completely misunderstood, or inputs being incorrect.

- **Score 2** – *Acceptable*  
  The core intent is preserved. Most words are correct, though there may be noticeable omissions, substitutions, or reordering.

- **Score 3** – *Near-Perfect Match*  
  The actual output closely matches the expected output, ignoring differences in punctuation, capitalization, spacing, and trivial formatting.
</Scoring Criteria>

<Guidance>
- Ignore differences in:
  - Punctuation (e.g., periods, commas, dashes)
  - Capitalization (e.g., "hello" vs. "HELLO")
  - Spacing (e.g., "icecream" vs. "ice cream")
- Treat numeric equivalents as equal (e.g., "2" == "two")
- Consider equivalent variants (e.g., "z" == "zed")
- Check for correctness of inputs, such as names, email addresses, spellings, phone numbers, etc. This should be a major error if at all wrong.
- If the actual transcription is in a different language, it is a major error, and should be scored as 1.
- For spelled out words, check for correctness of spelling, but differences in format (all caps, spaces, etc.) should be ignored.
- If a transcript is inaudible, this is not considered missing.
- If both actual and expected transcriptions are empty, score as 3 (perfect match)
</Guidance>

<Examples>

<Example Score 0 - Missing Transcription>
Expected:
Bom dia. De onde é que estar falando?

Actual:


Response:
{{
    "reason":"Missing Transcription",
    "score": 0,
}}
</Example>

<Example Score 1 - Really Bad but Not Missing>
Expected:
Good morning. Where are you calling from?

Actual:
Increase the credit card limit.

Response:
{{
    "reason":"the meaning is completely different",
    "score": 1,
}}
</Example>  
<Example Score 1 - Incorrect inputs>
Expected:
ID1593029

Actual:
I D one five nine three eight
</Example>

<Example Score 1 - Incorrect language>
Expected:
vale vale

Actual:
भले भले 
</Example>

<Example Score 2 - Acceptable>
Expected:
Attends une minute, attends une minute!

Actual:
Veuillez patienter un instant, s'il vous plaît.

Response:
{{
    "reason":"the meaning is preserved, but there are some substitutions",
    "score": 2,
}}
</Example>

<Example Score 3 - Near-Perfect Match>
Expected:
नमस्ते। आप कहाँ से बोल रहे हैं?

Actual:
नमस्ते... आप कहाँ से बोल रहे हैं?

Response:
{{
    "reason":"the utterances are semantically equivalent",
    "score": 3,
}}
</Example>

<Example Score 3 - Near-Perfect Match>
Expected:
Oi bom dia 

Actual:
Oi, bom dia!

Response:
{{
    "reason":"the utterances are semantically equivalent",
    "score": 3,
}}
</Example>
</Examples>

<Input>

These are the input transcripts, do not use any of the information from the examples to score the output.

Expected:
{gold_transcript}

Actual (this maybe empty, according to the scoring criteria):
{llm_transcript}
</Input>
""".strip()

EXTRACT_INPUTS_PROMPT = """
<Task>
Extract all customer-provided input values from the call transcript. Inputs include full names, email addresses, phone numbers, postal addresses, policy or account identifiers, order numbers, dates of birth, and other literal identifiers that the caller provides. Transcribe the values cleanly, but not do change the values. 
For example, if a zip code is transcribed as "1 2 3 4 5", return "12345" as the value. If an email is transcribed as "peyton wells at yahoo dot com", return "peyton.wells@yahoo.com" as the value. Transcribe first and last names separately.
</Task>

<ReturnFormat>
Return strict JSON with a top-level key "inputs" that is an array of objects with fields "type" and "value". Types must be chosen from: ["name","email","phone","street","city","zip","id","dob","other"]. Do not add extra keys. Examples:
{{
  "inputs": [
    {{"type": "name", "value": "Peyton"}},
    {{"type": "name", "value": "Wells"}},
    {{"type": "email", "value": "peyton.wells@yahoo.com"}},
    {{"type": "street", "value": "400 Pine Street"}},
    {{"type": "city", "value": "Seattle"}},
    {{"type": "zip", "value": "98101"}},
    {{"type": "phone", "value": "+1 206 555 0199"}}
  ]
}}
</ReturnFormat>

<Input>
Transcript:
{transcript}
</Input>
""".strip()
