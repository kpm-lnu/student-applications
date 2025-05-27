#!/usr/bin/env python3
"""
main.py
----------------------------------------
Benchmark classical ElGamal vs. EC ElGamal:
  - Measure average encryption/decryption time (ms)
  - Scale total data size from 1 MB up to 1 GB
  - Compute percentage speed-ups of EC over classical
  - Produce and save plots and summary table

Dependencies:
    pip install pycryptodome matplotlib

Usage:
    python main.py
"""

import random
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
from Crypto.PublicKey import ECC
from Crypto.PublicKey.ECC import EccPoint

# ------------------------------------------------------------------------------
# Classical ElGamal parameters (2048-bit MODP group from RFC 3526)
# ------------------------------------------------------------------------------
MODP_PRIME_HEX = """
FFFFFFFF FFFFFFFF C90FDAA2 2168C234 C4C6628B 80DC1CD1
29024E08 8A67CC74 020BBEA6 3B139B22 514A0879 8E3404DD
EF9519B3 CD3A431B 302B0A6D F25F1437 4FE1356D 6D51C245
E485B576 625E7EC6 F44C42E9 A63A3620 FFFFFFFF FFFFFFFF
"""
P_INT = int("".join(MODP_PRIME_HEX.split()), 16)
G_INT = 2

# ------------------------------------------------------------------------------
# EC ElGamal parameters (NIST P-256)
# ------------------------------------------------------------------------------
CURVE_NAME = "P-256"
_CURVE     = ECC._curves[CURVE_NAME]
G_ECC      = _CURVE.G
ORDER_ECC  = int(_CURVE.order)
P_CURVE    = int(_CURVE.p)


# ------------------------------------------------------------------------------
# Classical ElGamal: keygen, encrypt, decrypt
# ------------------------------------------------------------------------------
def keygen_classical() -> Tuple[int, int]:
    """
    Generate a classical ElGamal key pair.
    Returns (secret_key, public_key).
    """
    sk = random.randrange(2, P_INT - 2)
    pk = pow(G_INT, sk, P_INT)
    return sk, pk


def enc_classical(m: int, pk: int) -> Tuple[int, int]:
    """
    Encrypt integer message m under classical ElGamal.
    Returns ciphertext (c1, c2).
    """
    k  = random.randrange(2, P_INT - 2)
    c1 = pow(G_INT, k, P_INT)
    s  = pow(pk, k, P_INT)
    c2 = (m * s) % P_INT
    return c1, c2


def dec_classical(c1: int, c2: int, sk: int) -> int:
    """
    Decrypt classical ElGamal ciphertext (c1, c2).
    Returns original message m.
    """
    s     = pow(c1, sk, P_INT)
    s_inv = pow(s, -1, P_INT)
    return (c2 * s_inv) % P_INT


# ------------------------------------------------------------------------------
# EC ElGamal class
# ------------------------------------------------------------------------------
class ECElGamal:
    """
    EC ElGamal on NIST P-256 curve.
    Uses naive scalar-to-point encoding M = m * G for demonstration only.
    """
    def __init__(self):
        key = ECC.generate(curve=CURVE_NAME)
        self.sk = int(key.d)
        Q     = key.pointQ
        self.qx, self.qy = int(Q.x), int(Q.y)

    @staticmethod
    def _neg(point: EccPoint) -> EccPoint:
        """Return elliptic-curve point negation."""
        return EccPoint(int(point.x), (-int(point.y)) % P_CURVE, curve=CURVE_NAME)

    def encrypt(self, m: int) -> Tuple[EccPoint, EccPoint]:
        """
        Encrypt integer m with EC ElGamal.
        Returns (C1, C2).
        """
        M  = m * G_ECC
        k  = random.randrange(1, ORDER_ECC - 1)
        C1 = k * G_ECC
        Q  = EccPoint(self.qx, self.qy, curve=CURVE_NAME)
        S  = k * Q
        C2 = M + S
        return C1, C2

    def decrypt(self, C1: EccPoint, C2: EccPoint) -> EccPoint:
        """
        Decrypt EC ElGamal ciphertext (C1, C2).
        Returns point M = m*G.
        """
        S   = self.sk * C1
        neg = self._neg(S)
        return C2 + neg


# ------------------------------------------------------------------------------
# Utility: generate random integer messages < modulus
# ------------------------------------------------------------------------------
def generate_messages(count: int, msg_size_bytes: int, modulus: int) -> List[int]:
    """Generate 'count' random integers of 'msg_size_bytes' bytes reduced mod 'modulus'."""
    return [
        random.getrandbits(msg_size_bytes * 8) % (modulus - 2) + 1
        for _ in range(count)
    ]


# ------------------------------------------------------------------------------
# Benchmark runner
# ------------------------------------------------------------------------------
def benchmark(
    sizes_bytes: List[int],
    msg_size_bytes: int = 4 * 1024
) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    """
    For each total size in 'sizes_bytes', compute:
      - N = total_size // msg_size_bytes
      - avg classical encrypt ms, avg classical decrypt ms
      - avg EC encrypt ms, avg EC decrypt ms
    Returns (size_GB, enc_class, dec_class, enc_ec, dec_ec).
    """
    size_gb = [tb / 1024**3 for tb in sizes_bytes]
    enc_class, dec_class, enc_ec, dec_ec = [], [], [], []

    for total in sizes_bytes:
        N = max(1, total // msg_size_bytes)

        # Classical ElGamal
        sk, pk       = keygen_classical()
        msgs_c       = generate_messages(N, msg_size_bytes, P_INT)
        t0           = time.perf_counter()
        c_c          = [enc_classical(m, pk) for m in msgs_c]
        enc_class.append((time.perf_counter() - t0) * 1000 / N)

        t0           = time.perf_counter()
        _            = [dec_classical(c1, c2, sk) for c1, c2 in c_c]
        dec_class.append((time.perf_counter() - t0) * 1000 / N)

        # EC ElGamal
        ec           = ECElGamal()
        msgs_e       = generate_messages(N, msg_size_bytes, ORDER_ECC)
        t0           = time.perf_counter()
        c_e          = [ec.encrypt(m) for m in msgs_e]
        enc_ec.append((time.perf_counter() - t0) * 1000 / N)

        t0           = time.perf_counter()
        _            = [ec.decrypt(C1, C2) for C1, C2 in c_e]
        dec_ec.append((time.perf_counter() - t0) * 1000 / N)

    return size_gb, enc_class, dec_class, enc_ec, dec_ec


# ------------------------------------------------------------------------------
# Plotting helper
# ------------------------------------------------------------------------------
def plot_times(
    x: List[float],
    y1: List[float],
    y2: List[float],
    xlabel: str,
    ylabel: str,
    title: str,
    labels: Tuple[str, str],
    filename: str
) -> None:
    """Plot two series y1, y2 vs x and save to filename."""
    plt.figure()
    plt.plot(x, y1, marker="o", label=labels[0])
    plt.plot(x, y2, marker="o", label=labels[1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


# ------------------------------------------------------------------------------
# Main entrypoint
# ------------------------------------------------------------------------------
def main():
    # Define total sizes: 1 MB, 10 MB, 100 MB, 500 MB, 1 GB
    TOTAL_SIZES = [1, 10, 100, 500, 1024]
    sizes_bytes = [mb * 1024**2 for mb in TOTAL_SIZES]

    size_gb, enc_c, dec_c, enc_e, dec_e = benchmark(sizes_bytes)

    # Compute speed-ups (%)
    speedup_enc = [(c - e) / c * 100 for c, e in zip(enc_c, enc_e)]
    speedup_dec = [(c - e) / c * 100 for c, e in zip(dec_c, dec_e)]

    # Plot results
    plot_times(
        size_gb, enc_c, enc_e,
        xlabel="Total data size (GB)",
        ylabel="Avg encryption time (ms)",
        title="Encryption time vs total size (4 KB/msg)",
        labels=("Classical Encrypt", "EC Encrypt"),
        filename="encryption.png"
    )
    plot_times(
        size_gb, dec_c, dec_e,
        xlabel="Total data size (GB)",
        ylabel="Avg decryption time (ms)",
        title="Decryption time vs total size (4 KB/msg)",
        labels=("Classical Decrypt", "EC Decrypt"),
        filename="decryption.png"
    )

    # Print summary table
    print(f"{'Size (GB)':>8} | {'ClEnc (ms)':>10} | {'ECEnc (ms)':>10} | {'ΔEncrypt (%)':>12}")
    for gb, c, e, s in zip(size_gb, enc_c, enc_e, speedup_enc):
        print(f"{gb:8.3f} | {c:10.2f} | {e:10.2f} | {s:12.1f}")

    print(f"\n{'Size (GB)':>8} | {'ClDec (ms)':>10} | {'ECDec (ms)':>10} | {'ΔDecrypt (%)':>12}")
    for gb, c, e, s in zip(size_gb, dec_c, dec_e, speedup_dec):
        print(f"{gb:8.3f} | {c:10.2f} | {e:10.2f} | {s:12.1f}")


if __name__ == "__main__":
    main()
