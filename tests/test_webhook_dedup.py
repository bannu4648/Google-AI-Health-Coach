from backend.health_coach.core import database


def test_claim_whatsapp_message_allows_first_claim_only():
    message_id = "wamid.test-dedup-claim"
    assert database.claim_whatsapp_message(message_id) is True
    assert database.claim_whatsapp_message(message_id) is False


def test_claim_whatsapp_message_without_id_uses_body_hash():
    body_hash = "abc123deadbeef"
    assert database.claim_whatsapp_message(None, body_hash=body_hash) is True
    assert database.claim_whatsapp_message(None, body_hash=body_hash) is False


def test_record_whatsapp_reply_roundtrip():
    message_id = "wamid.test-reply-roundtrip"
    database.record_whatsapp_reply(
        message_id,
        reply_text="hello",
        send_status="sent",
        phone="85200000000",
    )
    row = database.get_whatsapp_reply(message_id)
    assert row is not None
    assert row["reply_text"] == "hello"
    assert row["send_status"] == "sent"
