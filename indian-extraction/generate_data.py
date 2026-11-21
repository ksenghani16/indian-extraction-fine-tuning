"""
generate_data.py - builds the synthetic dataset.

For every example we:
  1. invent fake values (names, amounts, dates ...)
  2. write them into a document using a random template
  3. remember the correct JSON ("gold") - we know it because WE chose the values
  4. optionally add noise (typos, OCR errors, odd line breaks)

Run:  python generate_data.py --n_train 3000
"""
import argparse, datetime, json, os, random
from faker import Faker
from schemas import SCHEMAS, TRAIN_SCHEMAS, HELDOUT_SCHEMAS, build_prompt
from noise import add_noise

# ---------------------------------------------------------------- helpers
MERCHANTS = ["SWIGGY", "ZOMATO", "AMAZON PAY", "BIGBASKET", "IRCTC", "UBER INDIA",
             "RELIANCE SMART", "DMART", "APOLLO PHARMACY", "BOOKMYSHOW", "OLA CABS"]
CITY_STATE = [("Mumbai", "Maharashtra"), ("Pune", "Maharashtra"), ("Delhi", "Delhi"),
              ("Bengaluru", "Karnataka"), ("Chennai", "Tamil Nadu"), ("Hyderabad", "Telangana"),
              ("Kolkata", "West Bengal"), ("Ahmedabad", "Gujarat"), ("Jaipur", "Rajasthan"),
              ("Lucknow", "Uttar Pradesh"), ("Kochi", "Kerala"), ("Nagpur", "Maharashtra")]
STREETS = ["MG Road", "Station Road", "Linking Road", "Gandhi Nagar Main Road", "12th Cross",
           "Temple Street", "Nehru Marg", "Park Avenue Road", "Residency Road", "Lake View Road"]
LOCALITIES = ["Andheri West", "Koramangala", "Kothrud", "Salt Lake", "Banjara Hills", "Adyar",
              "Vastrapur", "Malviya Nagar", "Gomti Nagar", "Indiranagar", "Dadar East", "Viman Nagar"]
LANDMARKS = ["Near Shiv Mandir", "Opp. City Mall", "Behind SBI Bank", "Near Ganesh Temple",
             "Next to Metro Station", "Near Hanuman Chowk", "Opp. Govt School"]
JOB_TITLES = ["Software Engineer", "Data Analyst", "Backend Developer", "QA Engineer",
              "DevOps Engineer", "Machine Learning Engineer", "Business Analyst", "UI Designer"]
SKILLS = ["Python", "Java", "SQL", "React", "AWS", "Docker", "Excel", "Tableau", "Kubernetes",
          "Node.js", "Django", "PyTorch", "Git", "Power BI"]
PROVIDERS = ["MSEDCL", "BESCOM", "Tata Power", "Adani Electricity", "TNEB", "BSES Rajdhani"]
STATIONS = ["Pune Jn", "Mumbai CSMT", "New Delhi", "Howrah Jn", "Chennai Central", "Nagpur",
            "Bhopal Jn", "Secunderabad Jn", "Jaipur", "Ahmedabad Jn", "Lucknow NR"]
STATE_CODES = ["27", "29", "07", "33", "36", "19", "24", "08", "09", "32"]


def inr(x: float) -> str:
    """1234567.5 -> '12,34,567.50' (Indian digit grouping)."""
    whole, frac = f"{x:.2f}".split(".")
    if len(whole) > 3:
        head, tail, groups = whole[:-3], whole[-3:], []
        while len(head) > 2:
            groups.insert(0, head[-2:]); head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups + [tail])
    return f"{whole}.{frac}"


def money(rng, x: float) -> str:
    return inr(x) if rng.random() < 0.5 else f"{x:.2f}"


def rand_date(rng):
    return datetime.date(2024, 1, 1) + datetime.timedelta(days=rng.randint(0, 1000))


def fmt_date(rng, d: datetime.date) -> str:
    return d.strftime(rng.choice(["%d-%b-%y", "%d/%m/%Y", "%d-%m-%Y", "%d %b %Y"]))


def gstin(rng) -> str:
    """Well-formed but FAKE GSTIN (state code + PAN-like + entity + Z + check char)."""
    L, D = "ABCDEFGHJKLMNPRSTUVWXYZ", "0123456789"
    pan = "".join(rng.choice(L) for _ in range(5)) + "".join(rng.choice(D) for _ in range(4)) + rng.choice(L)
    return rng.choice(STATE_CODES) + pan + str(rng.randint(1, 9)) + "Z" + rng.choice(D + L)


def company(rng, fake):
    return f"{fake.last_name()} {rng.choice(['Traders', 'Enterprises', 'Industries', '& Sons', 'Solutions'])}"


# ------------------------------------------------------------- generators
# Each returns (document_text, gold_json_dict, rendered_values)
# rendered_values = exact strings that appear in the document (used for noise
# protection and for a sanity check that the answer is really in the text).

def gen_upi_sms(rng, fake):
    debited = rng.random() < 0.6
    direction = "debited" if debited else "credited"
    amount = round(rng.choice([rng.uniform(10, 999), rng.uniform(1000, 99999)]), 2)
    amt_s = money(rng, amount)
    amt_txt = rng.choice(["Rs.", "Rs ", "INR ", "₹"]) + amt_s
    last4 = f"{rng.randint(0, 9999):04d}"
    party = rng.choice(MERCHANTS) if debited else fake.name()
    d = rand_date(rng); d_s = fmt_date(rng, d)
    ref = str(rng.randint(10**11, 10**12 - 1)) if rng.random() < 0.8 else None
    bal = round(rng.uniform(100, 200000), 2) if rng.random() < 0.4 else None
    bal_s = money(rng, bal) if bal is not None else None

    style = rng.randint(0, 2)
    if style == 0:
        prep = "from" if debited else "to"
        side = "to" if debited else "from"
        doc = f"{amt_txt} {direction} {prep} A/c XX{last4} on {d_s} {side} {party}."
        if ref: doc += f" UPI Ref {ref}."
        if bal is not None: doc += f" Avl Bal Rs.{bal_s}"
    elif style == 1:
        doc = (f"A/c XX{last4} {direction} for {amt_txt} on {d_s}. "
               f"{'Paid to' if debited else 'Received from'} {party}.")
        if ref: doc += f" Ref No: {ref}"
        if bal is not None: doc += f" Balance: INR {bal_s}"
    else:  # Hinglish
        verb = "debit" if debited else "credit"
        prep = "se" if debited else "mein"
        doc = f"Aapke A/c XX{last4} {prep} {amt_txt} {verb} hue {d_s} ko. {party}."
        if ref: doc += f" UPI ref {ref}."
        if bal is not None: doc += f" Bachi hui rashi Rs.{bal_s}"
    gold = {"amount_inr": amount, "direction": direction, "account_last4": last4,
            "counterparty": party, "date": d.isoformat(), "upi_ref": ref, "balance_inr": bal}
    rendered = [amt_s, last4, party, d_s] + ([ref] if ref else []) + ([bal_s] if bal_s else [])
    return doc, gold, rendered


def gen_gst_invoice(rng, fake):
    seller = company(rng, fake)
    s_gstin = gstin(rng)
    b_gstin = gstin(rng) if rng.random() < 0.7 else None
    inv_no = f"{rng.choice(['INV', 'GST', 'TI'])}/{rng.choice([2024, 2025, 2026])}/{rng.randint(1, 9999):04d}"
    d = rand_date(rng); d_s = fmt_date(rng, d)
    taxable = round(rng.uniform(500, 500000), 2)
    rate = rng.choice([5, 12, 18, 28])
    intra = rng.random() < 0.5
    if intra:
        cgst = sgst = round(taxable * rate / 200, 2); igst = None
        total = round(taxable + cgst + sgst, 2)
    else:
        cgst = sgst = None; igst = round(taxable * rate / 100, 2)
        total = round(taxable + igst, 2)
    lines = ["TAX INVOICE", seller, f"GSTIN: {s_gstin}",
             f"{rng.choice(['Invoice No', 'Inv No.', 'Bill No'])}: {inv_no}   Date: {d_s}"]
    rendered = [seller, s_gstin, inv_no, d_s]
    if b_gstin:
        lines.append(f"Buyer GSTIN: {b_gstin}"); rendered.append(b_gstin)
    t_s = money(rng, taxable); lines.append(f"Taxable Value: {t_s}"); rendered.append(t_s)
    if intra:
        c_s, s_s = money(rng, cgst), money(rng, sgst)
        lines += [f"CGST @{rate/2:g}%: {c_s}", f"SGST @{rate/2:g}%: {s_s}"]; rendered += [c_s, s_s]
    else:
        i_s = money(rng, igst)
        lines.append(f"IGST @{rate}%: {i_s}"); rendered.append(i_s)
    tot_s = money(rng, total)
    lines.append(f"{rng.choice(['Grand Total', 'Total Amount', 'Net Payable'])}: {tot_s}"); rendered.append(tot_s)
    gold = {"seller_name": seller, "seller_gstin": s_gstin, "buyer_gstin": b_gstin,
            "invoice_number": inv_no, "invoice_date": d.isoformat(), "taxable_value": taxable,
            "cgst": cgst, "sgst": sgst, "igst": igst, "total_amount": total}
    return "\n".join(lines), gold, rendered


def gen_address(rng, fake):
    house = rng.choice(["Flat ", "H.No. ", "Plot ", "B-", ""]) + str(rng.randint(1, 999))
    street = rng.choice(STREETS)
    landmark = rng.choice(LANDMARKS) if rng.random() < 0.6 else None
    locality = rng.choice(LOCALITIES)
    city, state = rng.choice(CITY_STATE)
    pin = str(rng.randint(100000, 899999))   # random, not necessarily a real pincode
    style = rng.randint(0, 2)
    parts = [house, street] + ([landmark] if landmark else []) + [locality, city, state]
    if style == 0:
        doc = ", ".join(parts) + f" - {pin}"
    elif style == 1:
        doc = "\n".join(parts) + f"\nPin: {pin}"
    else:
        doc = ", ".join(parts) + f" PIN {pin}"
    gold = {"house_no": house, "street": street, "landmark": landmark, "locality": locality,
            "city": city, "state": state, "pincode": pin}
    rendered = [house, street, locality, city, state, pin] + ([landmark] if landmark else [])
    return doc, gold, rendered


def gen_job_posting(rng, fake):
    title = rng.choice(JOB_TITLES); comp = company(rng, fake)
    city = rng.choice(CITY_STATE)[0]
    lo = rng.randint(0, 6); hi = lo + rng.randint(1, 4)
    has_sal = rng.random() < 0.7
    s_lo = rng.randint(3, 20) if has_sal else None
    s_hi = s_lo + rng.randint(2, 10) if has_sal else None
    skills = rng.sample(SKILLS, rng.randint(2, 4))
    if rng.random() < 0.5:
        doc = f"Hiring: {title} | {comp}, {city}\nExperience: {lo}-{hi} years\n"
        if has_sal: doc += f"Salary: {s_lo}-{s_hi} LPA\n"
        doc += "Skills: " + ", ".join(skills)
    else:
        doc = f"{comp} is looking for a {title} in {city}. Candidates with {lo} to {hi} yrs experience can apply."
        if has_sal: doc += f" CTC {s_lo} to {s_hi} LPA."
        doc += " Must know " + " / ".join(skills) + "."
    gold = {"job_title": title, "company": comp, "location": city, "experience_min_years": lo,
            "experience_max_years": hi, "salary_min_lpa": s_lo, "salary_max_lpa": s_hi, "skills": skills}
    return doc, gold, [title, comp, city] + skills


def gen_utility_bill(rng, fake):   # HELD-OUT schema
    prov = rng.choice(PROVIDERS); cons = str(rng.randint(10**11, 10**12 - 1))
    units = rng.randint(20, 900); amt = round(units * rng.uniform(4, 9), 2)
    d = rand_date(rng); d_s = fmt_date(rng, d); amt_s = money(rng, amt)
    if rng.random() < 0.5:
        doc = (f"{prov} ELECTRICITY BILL\nConsumer No: {cons}\nUnits Consumed: {units}\n"
               f"Amount Payable: Rs.{amt_s}\nDue Date: {d_s}")
    else:
        doc = (f"Dear customer, your {prov} bill for consumer {cons} ({units} units) is Rs.{amt_s}. "
               f"Please pay by {d_s}.")
    gold = {"provider": prov, "consumer_number": cons, "units_consumed": units,
            "amount_due": amt, "due_date": d.isoformat()}
    return doc, gold, [prov, cons, str(units), amt_s, d_s]


def gen_train_ticket(rng, fake):   # HELD-OUT schema
    pnr = str(rng.randint(10**9, 10**10 - 1)); name = fake.name()
    a, b = rng.sample(STATIONS, 2); d = rand_date(rng); d_s = fmt_date(rng, d)
    fare = float(rng.randint(150, 4500)); fare_s = f"{fare:.2f}"
    if rng.random() < 0.5:
        doc = f"PNR: {pnr}\nPassenger: {name}\n{a} to {b}\nDOJ: {d_s}\nTotal Fare: Rs {fare_s}"
    else:
        doc = f"Booking confirmed! PNR {pnr}. {name} travelling {a} -> {b} on {d_s}. Fare paid Rs.{fare_s}"
    gold = {"pnr": pnr, "passenger_name": name, "from_station": a, "to_station": b,
            "travel_date": d.isoformat(), "fare": fare}
    return doc, gold, [pnr, name, a, b, d_s, fare_s]


GENERATORS = {"upi_sms": gen_upi_sms, "gst_invoice": gen_gst_invoice, "address": gen_address,
              "job_posting": gen_job_posting, "utility_bill": gen_utility_bill,
              "train_ticket": gen_train_ticket}


# ---------------------------------------------------------------- dataset
def make_record(name, rng, fake, noise_level):
    for _ in range(30):
        doc, gold, rendered = GENERATORS[name](rng, fake)
        doc = add_noise(doc, noise_level, rendered, rng)
        # sanity check: every value we claim is in the text really is there
        if all(r in doc for r in rendered):
            break
    else:
        return None
    schema = SCHEMAS[name]
    prompt = build_prompt(schema, doc)
    completion = json.dumps(gold, ensure_ascii=False)
    return {"schema_name": name, "noise": noise_level, "document": doc, "gold": gold,
            "prompt": prompt, "completion": completion}


def make_split(n, schema_names, seed, banned_docs=frozenset()):
    rng = random.Random(seed); fake = Faker("en_IN"); fake.seed_instance(seed)
    records, seen = [], set()
    while len(records) < n:
        name = schema_names[len(records) % len(schema_names)]
        level = rng.choices([0, 1, 2], weights=[4, 4, 2])[0]
        rec = make_record(name, rng, fake, level)
        if rec is None or rec["document"] in seen or rec["document"] in banned_docs:
            continue
        seen.add(rec["document"]); records.append(rec)
    return records


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=3000)
    ap.add_argument("--n_val", type=int, default=200)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    train = make_split(a.n_train, TRAIN_SCHEMAS, a.seed)
    train_docs = {r["document"] for r in train}
    val = make_split(a.n_val, TRAIN_SCHEMAS, a.seed + 1, train_docs)
    test_in = make_split(a.n_test, TRAIN_SCHEMAS, a.seed + 2, train_docs)
    test_held = make_split(a.n_test, HELDOUT_SCHEMAS, a.seed + 3)

    only_sft = lambda rows: [{"prompt": r["prompt"], "completion": r["completion"]} for r in rows]
    write_jsonl(f"{a.out}/train_sft.jsonl", only_sft(train))   # used for training
    write_jsonl(f"{a.out}/val_sft.jsonl", only_sft(val))
    write_jsonl(f"{a.out}/train_full.jsonl", train)            # readable copy
    write_jsonl(f"{a.out}/test_in_schema.jsonl", test_in)      # synthetic test (same schemas)
    write_jsonl(f"{a.out}/test_heldout.jsonl", test_held)      # NEW schemas never trained on
    print(f"train={len(train)} val={len(val)} test_in={len(test_in)} test_heldout={len(test_held)} -> {a.out}/")
    print("\n--- sample document ---\n" + train[0]["document"] + "\n--- gold ---\n" + train[0]["completion"])
