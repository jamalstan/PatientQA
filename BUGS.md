# Bug report — Pretty Good AI scheduling line

This report contains human-reviewed findings from 11 synthetic-patient calls. Each finding links
to the complete timestamped transcript and representative audio. Repeated instances are grouped
instead of counted as separate bugs.

Severity reflects patient impact: **high** can produce an incorrect or unsafe outcome;
**medium** materially harms task completion or conversation quality.

## High severity

### 1. The agent accepts an incomplete callback number

**Call:** [call-008 transcript](deliverables/calls/call-008/transcript.txt) ·
[audio at 0:19](deliverables/calls/call-008/recording.mp3#t=19)

**Evidence:** The patient says, “I need to update my callback number to five five five three two
one zero.” The agent reads back `555-3210` and then says, “Your callback number is updated to
555-3210.”

**Impact:** A seven-digit number is not independently dialable and has no area code. Accepting it
without validation can make reminders and clinical follow-up unreachable. The agent should require
and confirm a complete ten-digit or E.164 number before saving it.

### 2. The agent gives fasting advice without knowing the ordered tests

**Call:** [call-008 transcript](deliverables/calls/call-008/transcript.txt) ·
[audio at 1:36](deliverables/calls/call-008/recording.mp3#t=96)

**Evidence:** After the patient only identifies “routine blood work,” the agent says, “For most
routine blood work, fasting for eight to 12 hours is often recommended, but it depends on the
specific test.”

**Impact:** Whether fasting is appropriate depends on the actual order and patient instructions.
Generic preparation advice may cause unnecessary fasting or conflict with medication and diabetes
management. The agent should avoid prescribing a fasting window until it can identify the tests,
then use clinic-approved instructions or escalate.

### 3. Correct provider identity is corrupted during a destructive action

**Call:** [research10-01 transcript](deliverables/calls/research10-01/transcript.txt) ·
[audio at 2:20](deliverables/calls/research10-01/recording.mp3#t=140)

**Evidence:** The agent initially identifies “Dr. Zbigniew Lukowski.” During cancellation it changes
that to “Dr. Zidnu Lukoski.” The patient corrects the name twice, but the final action statement is,
“I will cancel your appointment with Dr. Zigmew Lukowski.”

**Impact:** Cancellation is a destructive action. Proceeding while the provider identity remains
wrong risks modifying the wrong appointment and makes the confirmation unreliable. The agent
should preserve structured appointment data rather than regenerate names from speech, and it
should stop after a correction until every key field matches.

### 4. Long silent gaps make the line appear disconnected

**Representative calls:**

- [research10-03 at 1:22](deliverables/calls/research10-03/recording.mp3#t=82): the patient asks to
  reschedule, says “Hello?” at 1:35, and receives the next response at 2:19.
- [call-008 at 0:52](deliverables/calls/call-008/recording.mp3#t=52): after 33 seconds without an
  answer, the agent asks the waiting patient, “Are you still there?”
- [call-002 at 0:52](deliverables/calls/call-002/recording.mp3#t=52): a holiday availability lookup
  takes about 26 seconds with no progress acknowledgement.

**Impact:** These are not isolated pauses; response gaps above 15 seconds recur across the campaign,
with a worst observed gap of roughly 57 seconds after a request. Callers cannot distinguish this
from a dropped call and may repeat themselves, barge in, or hang up. The agent should immediately
acknowledge the request, provide periodic progress cues for slow tools, and retain the caller's turn
while the lookup completes.

## Medium severity

### 5. Repeated secondary questions are ignored

**Call:** [call-005 transcript](deliverables/calls/call-005/transcript.txt) ·
[audio at 2:17](deliverables/calls/call-005/recording.mp3#t=137)

**Evidence:** The patient asks which entrance to use three times. The responses are “The closest
times are 11:30,” “Would you like to try a different time or date,” and “Would you like to pick a
time now?”

**Impact:** The scheduling state overrides the patient's newest intent. A caller cannot pause a
booking flow to ask a practical question, and repeated clarification does not recover the
conversation. The agent should answer, explicitly say it lacks the information, or offer a transfer
before returning to the pending appointment.

### 6. Cancellation confirmation is incoherent and delayed

**Call:** [call-004 transcript](deliverables/calls/call-004/transcript.txt) ·
[audio at 0:33](deliverables/calls/call-004/recording.mp3#t=33)

**Evidence:** The patient says they want to cancel an existing appointment and twice asks for
confirmation. The agent responds “No problem,” then “You can scan the QR code at the booth,” then
“Is there anything else I can help you with?” It does not provide useful confirmation until much
later, when it says there are no upcoming appointments.

**Impact:** The caller has no clear moment at which the cancellation succeeded, and the unrelated
QR-code response makes the system state ambiguous. A cancellation flow should identify the exact
appointment, obtain confirmation, execute once, and immediately read back the result.

### 7. Reschedule options omit or lose required details

**Calls:** [call-006 transcript](deliverables/calls/call-006/transcript.txt) ·
[research10-02 transcript](deliverables/calls/research10-02/transcript.txt)

**Evidence:** In call-006, after the patient asks the agent to repeat the time, it only says, “Your
new appointment option is Tuesday, August 25th”; the patient has to supply “10:30 a.m.” In
research10-02, the caller accepts Thursday at 4:30, but the agent neither confirms availability nor
states that it is unavailable—it repeatedly asks for another day and never completes a read-back.

**Impact:** Date, time, provider, location, and action state are safety-critical booking fields.
Omitting them or silently losing an accepted counteroffer makes it easy for caller and system to
leave with different understandings. Every proposal and final confirmation should contain the
complete structured slot.

### 8. Basic identity recognition produces an unprofessional recovery

**Call:** [call-003 transcript](deliverables/calls/call-003/transcript.txt) ·
[audio at 0:23](deliverables/calls/call-003/recording.mp3#t=23)

**Evidence:** After the patient says “Iris Rojas,” the agent replies “Nope,” then later says, “You
already provided Widest Rojas.”

**Impact:** A name-recognition error is expected in voice systems; the failure is the recovery.
Rejecting the caller without a neutral clarification is confusing and disrespectful, and saving the
misheard identity can contaminate all later actions. The agent should ask the caller to repeat or
spell the name and explicitly confirm it before profile creation.

## Scope and caller-side notes

- All people, phone numbers, appointments, and medical details are synthetic and belong to the
  assessment environment.
- The long-silence scenario intentionally inserted a short patient hesitation. The reported agent
  gap is substantially longer and begins after an explicit reschedule request; the intentional
  perturbation is not itself classified as a bug.
- Calls were capped near three minutes. Findings only claim behavior audible in the linked evidence;
  an unfinished action at the cap is not treated as proof that the backend action failed.
- The repeated findings above were manually consolidated from deterministic timing checks and an
  evidence-constrained LLM review. No issue is based solely on a model-supplied timestamp.
