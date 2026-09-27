"""
--- Εξήγηση Κώδικα ---

Περιγραφή:
Παρέχει συναρτήσεις για την παρακολούθηση της χρήσης CPU και RAM
σε πραγματικό χρόνο κατά τη διάρκεια του inference.

Σημείωση για RAM:
Διαβάζουμε απευθείας από το /proc/meminfo αντί για psutil
γιατί το psutil.virtual_memory().percent εξαιρεί buffers/cache
από το "used", με αποτέλεσμα να αναφέρει χαμηλά ποσοστά ακόμα
και όταν το μοντέλο έχει φορτωθεί στη μνήμη.
Χρησιμοποιούμε (MemTotal - MemFree) / MemTotal που αντικατοπτρίζει
την πραγματική πίεση μνήμης συμπεριλαμβανομένων των model weights
που κρατά το Linux στο buff/cache.
"""

import psutil


def get_ram_usage() -> float:
    """
    Returns actual RAM pressure as a percentage.
    Reads from /proc/meminfo directly — same source as free -h.
    Uses (MemTotal - MemFree) / MemTotal which correctly includes
    memory held by model weights in Linux buff/cache.
    """
    with open('/proc/meminfo') as f:
        lines = f.readlines()
    meminfo = {}
    for line in lines:
        parts = line.split()
        meminfo[parts[0].rstrip(':')] = int(parts[1])  # values in kB

    total = meminfo['MemTotal']
    free  = meminfo['MemFree']
    return round((total - free) / total * 100, 2)


def get_cpu_usage() -> float:
    """
    Returns CPU usage percentage across all cores.
    Note: During inference one core runs near 100% so
    total across 4 cores shows ~25% — this is expected.
    """
    return psutil.cpu_percent(interval=0.1)