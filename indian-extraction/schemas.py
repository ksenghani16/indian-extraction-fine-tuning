"""
schemas.py - defines WHAT we want to extract from each document type.

A schema is a dict:  field_name -> short description of the expected value.
The schema is pasted INTO the prompt, so one model can handle many document
types. We keep some schemas out of training ("held-out") to test whether the
model really learned the skill of extraction instead of memorising templates.
"""
import json

SCHEMAS = {
    # ---------- schemas the model is TRAINED on ----------
    "upi_sms": {
        "amount_inr": "number",
        "direction": "one of: debited, credited",
        "account_last4": "string",
        "counterparty": "string",
        "date": "date YYYY-MM-DD",
        "upi_ref": "string or null",
        "balance_inr": "number or null",
    },
    "gst_invoice": {
        "seller_name": "string",
        "seller_gstin": "string",
        "buyer_gstin": "string or null",
        "invoice_number": "string",
        "invoice_date": "date YYYY-MM-DD",
        "taxable_value": "number",
        "cgst": "number or null",
        "sgst": "number or null",
        "igst": "number or null",
        "total_amount": "number",
    },
    "address": {
        "house_no": "string",
        "street": "string",
        "landmark": "string or null",
        "locality": "string",
        "city": "string",
        "state": "string",
        "pincode": "string",
    },
    "job_posting": {
        "job_title": "string",
        "company": "string",
        "location": "string",
        "experience_min_years": "number",
        "experience_max_years": "number",
        "salary_min_lpa": "number or null",
        "salary_max_lpa": "number or null",
        "skills": "list of strings",
    },
    # ---------- HELD-OUT schemas: never used in training ----------
    "utility_bill": {
        "provider": "string",
        "consumer_number": "string",
        "units_consumed": "number",
        "amount_due": "number",
        "due_date": "date YYYY-MM-DD",
    },
    "train_ticket": {
        "pnr": "string",
        "passenger_name": "string",
        "from_station": "string",
        "to_station": "string",
        "travel_date": "date YYYY-MM-DD",
        "fare": "number",
    },
}

TRAIN_SCHEMAS = ["upi_sms", "gst_invoice", "address", "job_posting"]
HELDOUT_SCHEMAS = ["utility_bill", "train_ticket"]


def build_prompt(schema: dict, document: str) -> str:
    """The exact text the model sees. Training and testing MUST use this same function."""
    return (
        "Extract the fields below from the document. "
        "Use null for any field that is not in the document. Return only JSON.\n"
        "### Schema\n" + json.dumps(schema, ensure_ascii=False) + "\n"
        "### Document\n" + document + "\n"
        "### JSON\n"
    )
