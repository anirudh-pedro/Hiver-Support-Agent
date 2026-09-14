# Golden Evaluation Set Labeling Guide

This guide establishes the labeling protocol for annotating the 220-thread **Golden Evaluation Set** (`data/golden/golden_set_template.csv`).

---

## 1. Ground Truth Philosophy & Core Principles

1. **Evaluate What *Should* Happen, Not What Historical Twitter Did**:
   - The historical replies in the dataset from `@AmazonHelp` were often generic, repetitive macros ("Please DM us your order number", "Reach us here: https://...").
   - Some historical replies were unhelpful, delayed, or inappropriate.
   - **Your task is to establish the ideal gold standard**: What would a **competent, empathetic, and efficient human support agent** do?
2. **100% Unbiased & Model-Blind**:
   - The dataset contains strictly unseen, fresh threads with all LLM predictions removed.
   - The 220 threads are randomly shuffled across categories to avoid sequential category bias.
3. **Intent vs. Escalation**:
   - **Intent** identifies *what the customer's issue is*.
   - **Escalation** determines *how the customer support system should operationalize the response* (autonomous AI agent vs. human specialist handoff).
4. **Primary Stated Issue vs. Drift**:
   - If a thread's issue drifts or a second distinct problem emerges after the opening message, label `gold_intent` based on the customer's primary/opening stated issue only. Note any secondary issue in `gold_reply_notes`, but do not let it change `gold_intent`.

---

## 2. Sampling Methodology & Stratum Partitioning

### How Strata Were Assigned to the ~60,500 Unlabeled Threads
To construct an evaluation set of 220 threads that mirrors the real-world intent distribution while oversampling rare classes, we needed to partition the 60,575 unseen threads into candidate strata *without using an LLM* (which would introduce circular model bias and hit API rate limits):

1. **Strict Exclusion Filter**:
   - All 300 threads from the preliminary taxonomy validation and the 60 spot-check threads were loaded into an exclusion set.
   - Any thread in this set was skipped, guaranteeing that the 220 golden threads are 100% unseen data.

2. **High-Precision Lexical Stratum Filters**:
   - Candidate pools were created by scanning the customer's opening message using high-precision lexical patterns with **strict priority ordering** (rare classes evaluated first so they are not subsumed by broader intents):
     - `scam_phishing_check` (priority 1): Phishing, scam, spoof, fake email/message, fake prize.
     - `account_access` (priority 2): Locked out, 2FA/OTP, password reset, hacked, sign-in failure.
     - `billing_dispute` (priority 3): Double charge, unauthorized fee, refund, cashback, subscription billing.
     - `order_product_problem` (priority 4): Defective, broken, damaged, wrong item, crushed box, app crash.
     - `service_complaint_vague` (priority 5): Poor customer service venting without a specific order/defect named.
     - `delivery_delay` (priority 6): Late, delayed, tracking stalled, where is my order, delivered-but-missing.
     - `product_info_question` (priority 7): Compatibility, specifications, release dates, warranties, feature inquiry.
     - `non_support` (priority 8): Praise, compliments, shout-outs, holiday greetings, banter.
     - `other` (priority 9): Real support topics outside the core 8 (e.g. Amazon Locker codes, trade-in program, gift registries, order cancellation status), with negative filtering against delivery, billing, defect, and login terms.

3. **Stratified Sampling Targets & Oversampling**:
   - Total sample size: **220 threads**.
   - **Rare classes oversampled to 14 each** (at least 12–15) to ensure statistical significance for per-class metrics: `account_access` (14), `scam_phishing_check` (14), `other` (14).
   - Remaining 178 slots allocated proportionally to empirical distribution:
     - `delivery_delay`: 66 (~35%)
     - `service_complaint_vague`: 38 (~20%)
     - `product_info_question`: 25 (~13%)
     - `order_product_problem`: 19 (~10%)
     - `non_support`: 19 (~10%)
     - `billing_dispute`: 11 (~6%)

4. **Deterministic Seeds & Blind Presentation**:
   - `SEED_STRATIFIED = 42`: Deterministically samples the required count from each candidate pool.
   - `SEED_SHUFFLE = 777`: Randomly shuffles the 220 selected threads across categories so human annotators are completely blind to candidate stratum blocks during labeling.
   - **Crucial note**: These lexical filters were used **only to guarantee diversity in the sample pool**. The true intent ground truth is 100% determined by your human annotation in `golden_set_template.csv`.

---

## 3. File Columns & Annotation Schema

In `data/golden/golden_set_template.csv`, annotate the 4 `gold_*` columns for each row:

| Column Header | Type | Valid Values | Description |
| :--- | :--- | :--- | :--- |
| `thread_id` | Integer | *(Read-only)* | Unique identifier for the Twitter conversation thread. |
| `full_thread_text` | Text | *(Read-only)* | Full chronological conversation with `[Customer]` and `[AmazonHelp]` turn labels. |
| `n_turns` | Integer | *(Read-only)* | Total number of tweets in the conversation thread. |
| `gold_intent` | Categorical | Exactly one of the 9 categories below | The primary customer intent of the opening issue. |
| `gold_escalate` | Categorical | `auto_handle` or `escalate` | Operational routing decision for an AI triage agent. |
| `gold_reason` | Free text | 1–2 concise sentences | Specific justification explaining the escalation or auto-handle decision. |
| `gold_reply_notes` | Free text | Bullet points / concise notes *(Optional)* | Key elements a correct, compliant agent response must cover. |

---

## 4. Locked Intent Taxonomy & Boundary Clarification Rules

The taxonomy comprises **8 core intent categories + "other"**:

| Intent Key | Definition & Scope |
| :--- | :--- |
| **`delivery_delay`** | Package is late, has not arrived by promised/guaranteed delivery date, carrier tracking is stalled, or package is marked as "delivered" but physically missing from doorstep/mailbox. |
| **`billing_dispute`** | Unexpected or unauthorized charges, incorrect promotional discounts/cashback, refund not received after return, duplicate credit card charges, or disputed Prime subscription/renewal fees. |
| **`order_product_problem`** | Wrong item shipped, defective/broken/damaged item, missing parts from package, sizing/color mismatch, **hardware/software app technical defects** (e.g. Fire TV app crashing, Prime Video stream error), or **concrete physical packaging defects** (e.g. crushed box, torn packaging, fragile item shipped in envelope). |
| **`account_access`** | Compromised/hacked account, locked out due to suspicious activity, forgotten password reset failures, Two-Factor Authentication (2FA) / OTP verification code delivery issues, or unable to sign in. |
| **`product_info_question`** | Factual inquiry about product specifications, hardware compatibility, release dates, restock timeline, warranty coverage, or general feature questions (inquiry only, not an existing broken product complaint). |
| **`service_complaint_vague`** | Customer venting frustration about poor customer service, rude phone/chat representatives, long hold times, or generally awful experience **ONLY when there is NO specific defect, order, or product named**. |
| **`scam_phishing_check`** | Customer asking if an email, SMS, phone call, or prize notification claiming to be from Amazon is legitimate or a phishing/spoof scam. |
| **`non_support`** | Praise, shout-outs, compliments, jokes, social media banter, or off-topic chatter with no active support problem. |
| **`other`** | Real support inquiry that fundamentally does not fit the 8 categories above (e.g. Amazon Locker access code issues, trade-in/sell-back questions, gift/wedding registry questions, corporate feedback, website navigation/layout bugs). |

### ⚠️ The Two Locked Boundary Rules

> [!IMPORTANT]
> **Rule 1: Concrete Defects vs. Vague Complaints**
> If a customer expresses anger, venting, or harsh emotion, but names a **concrete broken thing** (e.g., app crash, video playback failure, crushed packaging, defective device), classify as **`order_product_problem`**, **NOT** `service_complaint_vague`.
> `service_complaint_vague` is reserved *strictly* for complaints about service/process quality where no specific physical/software defect or order issue is identified.

> [!IMPORTANT]
> **Rule 2: Humor & Sarcasm vs. Non-Support**
> The presence of emojis (e.g. 😂, 📦, 🙄), sarcasm, irony, or jokes does **NOT** make a message `non_support` if it still describes a real delivery, defect, or billing issue (e.g., *"Thanks for sending me a completely flat pancake instead of my laptop box 😂"* $\rightarrow$ **`order_product_problem`**, not `non_support`).

---

## 5. Operational Routing: `auto_handle` vs. `escalate`

> [!TIP]
> **Full Thread Context for Escalation**:
> Base `gold_escalate` on the full thread context — including repeated broken promises, prior failed contacts, or escalating tone across turns — not solely on the opening message.

### When to label `auto_handle`:
The AI agent can safely resolve or guide the customer autonomously:
- Factual questions about policies, compatibility, release dates, or specifications.
- Guidance on how to track a package, locate carrier links, or submit a self-service return via the online Returns Center.
- Confirming that a suspicious email is phishing and providing the official report address (`stop-spoofing@amazon.com`).
- Closing or acknowledging positive praise / non-support chatter with brand gratitude.
- Directing a customer to standard self-service account recovery pages (if credentials/account are not actively compromised).

### When to label `escalate`:
A human specialist must intervene directly:
- **Financial actions**: Issuing refunds, processing fee reversals, investigating unauthorized credit card charges.
- **Security & Privacy**: Account takeover, hacked credentials, unauthorized password changes, sharing private PII.
- **Carrier / Logistics Escalation**: High-value missing packages marked delivered, driver misconduct, stolen packages.
- **Physical Defect Replacement**: Initiating immediate replacement for damaged expensive merchandise or hazardous items.
- **High Agitation / Churn Risk**: Furious customer threatening legal action, social media escalation, or closing long-time Prime membership after multiple agent failures.

---

## 6. Worked Examples by Field

### Worked Examples 1: `gold_intent`

1. **Example A**:
   - *Message*: *"@AmazonHelp My Fire TV stick app crashes every time I open Netflix or Prime Video. Black screen immediately. Fix this garbage!"*
   - **`gold_intent`**: `order_product_problem`
   - *Reasoning*: Concrete software app defect on a device, fits `order_product_problem` per Boundary Rule 1 despite venting tone.

2. **Example B**:
   - *Message*: *"@AmazonHelp Tracking says package was handed to resident at 2 PM, but I was at work and my front porch is empty. No package anywhere."*
   - **`gold_intent`**: `delivery_delay`
   - *Reasoning*: "Delivered but missing" package scenario.

3. **Example C**:
   - *Message*: *"Got an email from 'support-amazon-security.xyz' saying my account is suspended and to click a link to verify my SSN. Is this real?"*
   - **`gold_intent`**: `scam_phishing_check`
   - *Reasoning*: Customer verifying legitimacy of a suspicious communication.

4. **Example D**:
   - *Message*: *"Does the new Echo Dot 4th Gen support 5GHz Wi-Fi or only 2.4GHz?"*
   - **`gold_intent`**: `product_info_question`
   - *Reasoning*: Factual inquiry regarding technical specifications.

5. **Example E**:
   - *Message*: *"I dropped off my return package in Amazon Locker 'Pine' on 5th Ave yesterday, but my return status still says 'Awaiting Drop-off'. How do I refresh the locker status?"*
   - **`gold_intent`**: `other`
   - *Reasoning*: Amazon Locker operational issue; not an item defect or delivery delay.

---

### Worked Examples 2: `gold_escalate` & `gold_reason`

1. **Example A (Auto-Handle / Scam)**:
   - *Message*: *"Just received this text claiming I won a $500 Amazon voucher: http://bit.ly/fake. Is this from you guys?"*
   - **`gold_escalate`**: `auto_handle`
   - **`gold_reason`**: Public domain phishing inquiry. AI agent can immediately identify the scam, advise not clicking links, and direct user to report it to `stop-spoofing@amazon.com`.

2. **Example B (Auto-Handle / Product Info)**:
   - *Message*: *"Is the Kindle Paperwhite waterproof? Can I take it to the pool?"*
   - **`gold_escalate`**: `auto_handle`
   - **`gold_reason`**: Standard factual product specification that an AI agent can answer with official IPX8 rating details.

3. **Example C (Escalate / Billing Dispute)**:
   - *Message*: *"Why did Amazon charge my credit card $99.99 for Prime when I cancelled the free trial 3 days ago? Need this refunded now."*
   - **`gold_escalate`**: `escalate`
   - **`gold_reason`**: Involves financial transaction review and processing a monetary refund to the customer's payment method.

4. **Example D (Escalate / Account Security)**:
   - *Message*: *"My email and phone number were changed on my account without my permission. I'm locked out and orders are being placed right now!"*
   - **`gold_escalate`**: `escalate`
   - **`gold_reason`**: Active account takeover in progress; requires urgent security team freeze and identity verification.

5. **Example E (Escalate / Severe Service Failure)**:
   - *Message*: *"3 different chat agents promised me my baby's formula would arrive today. Now tracking says Friday. Absolutely unacceptable service!"*
   - **`gold_escalate`**: `escalate`
   - **`gold_reason`**: Critical essential item delayed with repeated agent miscommitments; human specialist required for re-dispatch or compensation.

---

### Worked Examples 3: `gold_reply_notes`

1. **For Delivery Delay (Missing Package)**:
   - **`gold_reply_notes`**: Empathize with delivery anxiety. Advise checking safe locations / neighbors. Instruct customer to send Order ID and address via secure DM (never public tweet). Offer replacement or refund if package is confirmed lost.

2. **For Damaged Item**:
   - **`gold_reply_notes`**: Apologize for damaged condition. Clarify that customer does not need to pay for return shipping. Provide direct link to Online Returns Center (`amazon.com/returns`) or offer DM assistance for instant replacement.

3. **For Phishing Report**:
   - **`gold_reply_notes`**: Confirm that Amazon will never request passwords or SSNs via email/SMS. Instruct not to click any links or provide personal info. Provide the official reporting forward address (`stop-spoofing@amazon.com`).

4. **For Account Lockout**:
   - **`gold_reply_notes`**: Acknowledge urgency of account lockout. Warn against sharing credentials publicly. Direct user to secure account recovery portal (`amazon.com/help/account-recovery`) or offer secure verification via official support phone/chat.

---

## 7. Practical Labeling Workflow

1. Open `data/golden/golden_set_template.csv` in Excel, LibreOffice Calc, or VS Code (with CSV Rainbow / Edit CSV extension).
2. For each row:
   - Read `full_thread_text` (observe customer opening problem and any brand clarification).
   - Enter `gold_intent` using the exact lower-case snake_case taxonomy keys.
   - Enter `gold_escalate` as either `auto_handle` or `escalate`.
   - Enter `gold_reason` explaining your decision in 1–2 sentences.
   - Optionally enter `gold_reply_notes` highlighting essential points for evaluating future response generation.
3. Save the completed file with UTF-8 encoding.
