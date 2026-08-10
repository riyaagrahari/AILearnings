from src.ledger import Ledger

def test_ledger_hash_chain(tmp_path):
    l=Ledger(tmp_path/'ledger.jsonl')
    l.append({'step':0,'batch_id':'b0'})
    l.append({'step':1,'batch_id':'b1'})
    assert l.verify_chain()
