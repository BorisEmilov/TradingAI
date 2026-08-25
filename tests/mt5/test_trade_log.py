import csv
import threading

from tradingai.mt5.trade_log import append_trade_event


def test_writes_header_once_and_appends_rows(tmp_path):
    path = tmp_path / "trades.csv"

    append_trade_event(path, event="APERTURA", ticket=1, symbol="EURUSD", price=1.1000)
    append_trade_event(path, event="CIERRE", ticket=1, symbol="EURUSD", price=1.1010, profit=50.0)

    with open(path) as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert rows[0]["event"] == "APERTURA"
    assert rows[1]["event"] == "CIERRE"
    assert rows[1]["profit"] == "50.0"


def test_concurrent_writes_dont_corrupt_or_interleave_rows(tmp_path):
    # Simula los 5 procesos de run_live.py escribiendo a la misma tabla a la vez.
    path = tmp_path / "trades.csv"
    n_threads = 5
    n_writes_per_thread = 20

    def _writer(thread_id: int) -> None:
        for i in range(n_writes_per_thread):
            append_trade_event(path, event="APERTURA", ticket=thread_id * 1000 + i, symbol=f"SYM{thread_id}")

    threads = [threading.Thread(target=_writer, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    with open(path) as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == n_threads * n_writes_per_thread
    tickets = {int(r["ticket"]) for r in rows}
    assert len(tickets) == n_threads * n_writes_per_thread  # ninguna fila se perdio o se corrompio
