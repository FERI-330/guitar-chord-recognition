"""
tests/test_fret_detectors.py

Regressziós teszt-szkript: "A rendszer lefut a régi és az új modullal is, hiba nélkül."

Futtatás:
    python tests/test_fret_detectors.py

Kimenet: PASS / FAIL sorok + összefoglaló táblázat.
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import PATHS
from src.fretboard import (
    FretDetectorInterface,
    GeometricFretDetector,
    IntensityFretDetector,
    run_v14_pipeline,
    _make_empty_fit,
)
from src.features import assemble_feature_vector, FEATURE_DIM

# ── ANSI színek (terminálhoz) ───────────────────────────────────────────────
_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_RESET  = "\033[0m"
_BOLD   = "\033[1m"


def _pass(msg: str) -> None:
    print(f"  {_GREEN}PASS{_RESET}  {msg}")


def _fail(msg: str) -> None:
    print(f"  {_RED}FAIL{_RESET}  {msg}")


def _info(msg: str) -> None:
    print(f"  {_YELLOW}INFO{_RESET}  {msg}")


# ──────────────────────────────────────────────────────────────────────────────
# SZEKCIÓ 1 – Osztály-szintű tesztek (kép nélkül)
# ──────────────────────────────────────────────────────────────────────────────

def test_abc_enforcement() -> bool:
    """FretDetectorInterface nem példányosítható."""
    try:
        FretDetectorInterface()
        _fail("FretDetectorInterface példányosítható – ABC nem érvényesül")
        return False
    except TypeError:
        _pass("FretDetectorInterface ABC nem példányosítható")
        return True


def test_required_keys_mock() -> bool:
    """Mindkét detektor kötelező kulcsokat ad vissza dummy kanonikus képen."""
    canon = np.zeros((80, 600, 3), dtype=np.uint8)
    for x in [50, 105, 165, 230, 302, 381]:
        import cv2
        cv2.line(canon, (x, 0), (x, 79), (200, 200, 200), 2)

    required = {"fit", "fret_xs_raw", "fret_xs_filt", "removed_pairs", "method"}
    fit_required = {"predicted_x", "coverage_ratio", "matched_frets",
                    "offset", "scale", "fit_method"}
    ok = True
    for cls, detector in [("GeometricFretDetector", GeometricFretDetector()),
                           ("IntensityFretDetector", IntensityFretDetector())]:
        r = detector.detect(canon)
        missing = required - r.keys()
        if missing:
            _fail(f"{cls}: hiányzó kulcsok: {missing}")
            ok = False
        else:
            fit_missing = fit_required - r["fit"].keys()
            if fit_missing:
                _fail(f"{cls}.fit: hiányzó kulcsok: {fit_missing}")
                ok = False
            else:
                _pass(f"{cls} kötelező kulcsok: OK")
    return ok


def test_fit_key_parity_mock() -> bool:
    """A két detektor fit dict-je azonos kulcshalmazt tartalmaz."""
    canon = np.zeros((80, 600, 3), dtype=np.uint8)
    r_geo = GeometricFretDetector().detect(canon)
    r_int = IntensityFretDetector().detect(canon)
    keys_geo = set(r_geo["fit"].keys())
    keys_int = set(r_int["fit"].keys())
    if keys_geo != keys_int:
        _fail(f"Eltérő fit kulcsok  csak-geo={keys_geo - keys_int}  csak-int={keys_int - keys_geo}")
        return False
    _pass(f"fit dict kulcsparítás: OK  ({len(keys_geo)} kulcs)")
    return True


def test_intensity_profile_shape() -> bool:
    """IntensityFretDetector.gradient_profile() visszaad 1D, CANONICAL_W hosszú tömböt."""
    canon = np.random.randint(0, 255, (80, 600, 3), dtype=np.uint8)
    det = IntensityFretDetector()
    profile = det.gradient_profile(canon)
    if profile.shape != (600,):
        _fail(f"Várt shape (600,), kapott {profile.shape}")
        return False
    if not (0.0 <= profile.min() and profile.max() <= 1.01):
        _fail(f"Profil nem normalizált: min={profile.min():.3f}  max={profile.max():.3f}")
        return False
    _pass(f"IntensityFretDetector.gradient_profile: shape={profile.shape}  max={profile.max():.3f}")
    return True


def test_empty_fit_structure() -> bool:
    """_make_empty_fit() azonos kulcsokat tartalmaz mint step8 kimenetele."""
    from src.geometry import step8_fit_fret_rule
    real_empty = step8_fit_fret_rule([])  # kevés pont → empty visszatérés
    our_empty = _make_empty_fit()
    # Eltérés: 'method' kulcs amit mi adunk hozzá
    expected_extra = {"method"}
    diff = set(our_empty.keys()) - set(real_empty.keys()) - expected_extra
    if diff:
        _fail(f"_make_empty_fit extra kulcsok (váratlan): {diff}")
        return False
    missing = set(real_empty.keys()) - set(our_empty.keys())
    if missing:
        _fail(f"_make_empty_fit hiányzó kulcsok: {missing}")
        return False
    _pass("_make_empty_fit struktúra kompatibilis step8_fit_fret_rule-lal")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# SZEKCIÓ 2 – Valós képeken futtatott tesztek
# ──────────────────────────────────────────────────────────────────────────────

def _load_sample_entries(n: int = 5) -> list[dict]:
    """Tesztképek betöltése a manifest train splitjéből."""
    manifest_path = PATHS["manifest"]
    if not manifest_path.exists():
        return []
    df = pd.read_csv(manifest_path)
    train_df = df[df["split"] == "train"].copy()
    # 1-1 kép minden osztályból, max n darab
    sample = (
        train_df.sort_values("class")
        .groupby("class", group_keys=False)
        .apply(lambda g: g.sample(1, random_state=42))
        .head(n)
        .reset_index(drop=True)
    )
    return [
        {"path": row["path"], "class": row["class"], "filename": row["filename"]}
        for _, row in sample.iterrows()
    ]


def _run_both_detectors(entry: dict) -> tuple[dict, dict]:
    """Futtat egy képen mindkét detektort, visszaad (geo_result, int_result)."""
    geo_det = GeometricFretDetector()
    int_det = IntensityFretDetector()
    r_geo = run_v14_pipeline(entry, fret_detector=geo_det)
    r_int = run_v14_pipeline(entry, fret_detector=int_det)
    return r_geo, r_int


def test_pipeline_runs_without_exception(entries: list[dict]) -> bool:
    """run_v14_pipeline nem dob kivételt mindkét detektorral."""
    if not entries:
        _info("Nem található manifest – teszt kihagyva")
        return True
    ok = True
    for entry in entries:
        fname = entry.get("filename", str(entry["path"]))[:35]
        for det_name, det in [("geo", GeometricFretDetector()),
                               ("int", IntensityFretDetector())]:
            try:
                r = run_v14_pipeline(entry, fret_detector=det)
                assert isinstance(r, dict), "Nem dict visszatérési értéke"
                assert "ok" in r
                assert "fit" in r or r.get("invalid_reason") is not None
            except Exception as exc:
                _fail(f"Kivétel ({det_name}) – {fname}: {exc}")
                traceback.print_exc()
                ok = False
    if ok:
        _pass(f"run_v14_pipeline nem dob kivételt ({len(entries)} kép × 2 detektor)")
    return ok


def test_result_dict_key_parity(entries: list[dict]) -> bool:
    """Mindkét detektor azonos kulcsokat produkál a result dict-ben."""
    if not entries:
        _info("Nem található manifest – teszt kihagyva")
        return True
    ok = True
    for entry in entries:
        fname = entry.get("filename", "?")[:30]
        try:
            r_geo, r_int = _run_both_detectors(entry)
            keys_geo = set(r_geo.keys())
            keys_int = set(r_int.keys())
            # Elfogadott eltérések: intensity_profile (csak IntensityFretDetector)
            extra_int = keys_int - keys_geo - {"intensity_profile"}
            missing_int = keys_geo - keys_int - {"fret_detector_method"}
            if extra_int or missing_int:
                _fail(f"Kulcs eltérés [{fname}]  "
                      f"extra_int={extra_int}  missing_int={missing_int}")
                ok = False
        except Exception as exc:
            _fail(f"Hiba [{fname}]: {exc}")
            ok = False
    if ok:
        _pass(f"result dict kulcs-parítás: OK  ({len(entries)} kép)")
    return ok


def test_feature_vector_compatible(entries: list[dict]) -> bool:
    """assemble_feature_vector nem dob kivételt egyik detektortól sem."""
    if not entries:
        _info("Nem található manifest – teszt kihagyva")
        return True
    ok = True
    for entry in entries:
        fname = entry.get("filename", "?")[:30]
        try:
            r_geo, r_int = _run_both_detectors(entry)
            feat_geo = assemble_feature_vector(r_geo)
            feat_int = assemble_feature_vector(r_int)
            if feat_geo.shape != (FEATURE_DIM,):
                _fail(f"Geo feature shape: {feat_geo.shape} != ({FEATURE_DIM},)")
                ok = False
            if feat_int.shape != (FEATURE_DIM,):
                _fail(f"Int feature shape: {feat_int.shape} != ({FEATURE_DIM},)")
                ok = False
            if np.any(np.isnan(feat_geo)) or np.any(np.isinf(feat_geo)):
                _fail(f"Geo feature NaN/Inf [{fname}]")
                ok = False
            if np.any(np.isnan(feat_int)) or np.any(np.isinf(feat_int)):
                _fail(f"Int feature NaN/Inf [{fname}]")
                ok = False
        except Exception as exc:
            _fail(f"assemble_feature_vector hiba [{fname}]: {exc}")
            traceback.print_exc()
            ok = False
    if ok:
        _pass(f"assemble_feature_vector: OK  (mindkét detektor, {len(entries)} kép)")
    return ok


def test_detector_method_label(entries: list[dict]) -> bool:
    """result['fret_detector_method'] helyes értéket tartalmaz."""
    if not entries:
        _info("Nem található manifest – teszt kihagyva")
        return True
    ok = True
    for entry in entries[:3]:
        r_geo, r_int = _run_both_detectors(entry)
        # fret_detector_method == "none" is expected when pipeline fails before step 12
        expected_geo = "geometric" if r_geo.get("ok") else "none"
        expected_int = "intensity" if r_int.get("ok") else "none"
        if r_geo.get("fret_detector_method") != expected_geo:
            _fail(f"geo method label: '{r_geo.get('fret_detector_method')}' != '{expected_geo}'")
            ok = False
        if r_int.get("fret_detector_method") != expected_int:
            _fail(f"int method label: '{r_int.get('fret_detector_method')}' != '{expected_int}'")
            ok = False
    if ok:
        _pass("fret_detector_method label: OK")
    return ok


# ──────────────────────────────────────────────────────────────────────────────
# SZEKCIÓ 3 – Összehasonlítás és statisztika
# ──────────────────────────────────────────────────────────────────────────────

def compare_detectors_on_images(entries: list[dict]) -> None:
    """Összehasonlító táblázat nyomtatása: ok-ráta, coverage_ratio, fret szám."""
    if not entries:
        return
    rows = []
    for entry in entries:
        fname = entry.get("filename", "?")[:25]
        cls = entry.get("class", "?")
        try:
            r_geo, r_int = _run_both_detectors(entry)
            fit_geo = r_geo.get("fit") or {}
            fit_int = r_int.get("fit") or {}
            rows.append({
                "file":      fname,
                "class":     cls,
                "geo_ok":    r_geo.get("ok", False),
                "int_ok":    r_int.get("ok", False),
                "geo_cov":   fit_geo.get("coverage_ratio", 0.0),
                "int_cov":   fit_int.get("coverage_ratio", 0.0),
                "geo_nfret": len(fit_geo.get("predicted_x", {})),
                "int_nfret": len(fit_int.get("predicted_x", {})),
                "geo_score": fit_geo.get("score", 0.0),
                "int_score": fit_int.get("score", 0.0),
            })
        except Exception as exc:
            rows.append({"file": fname, "class": cls, "error": str(exc)})

    df = pd.DataFrame(rows)
    print()
    print(f"{'Fájl':<26} {'Cls':<4} {'geo_ok':<7} {'int_ok':<7} "
          f"{'geo_cov':<9} {'int_cov':<9} {'geo_n':<7} {'int_n':<7} "
          f"{'geo_sc':<8} {'int_sc':<8}")
    print("-" * 95)
    for _, r in df.iterrows():
        if "error" in r:
            print(f"{r['file']:<26} {r['class']:<4}  HIBA: {r['error']}")
            continue
        print(f"{r['file']:<26} {r['class']:<4} "
              f"{'✓' if r['geo_ok'] else '✗':<7} {'✓' if r['int_ok'] else '✗':<7} "
              f"{r['geo_cov']:.3f}{'':>4} {r['int_cov']:.3f}{'':>4} "
              f"{r['geo_nfret']:<7} {r['int_nfret']:<7} "
              f"{r['geo_score']:.2f}{'':>4} {r['int_score']:.2f}")

    ok_df = df[~df.get("error", pd.Series(dtype=str)).notna()] if "error" in df else df
    if len(ok_df):
        print()
        print(f"  Összesítés ({len(ok_df)} kép):")
        print(f"    geo ok-rate:     {ok_df['geo_ok'].mean()*100:.1f}%")
        print(f"    int ok-rate:     {ok_df['int_ok'].mean()*100:.1f}%")
        print(f"    geo avg cov:     {ok_df['geo_cov'].mean():.3f}")
        print(f"    int avg cov:     {ok_df['int_cov'].mean():.3f}")
        print(f"    geo avg fretek:  {ok_df['geo_nfret'].mean():.1f}")
        print(f"    int avg fretek:  {ok_df['int_nfret'].mean():.1f}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import warnings
    warnings.filterwarnings("ignore")

    print(f"\n{_BOLD}=== Fret Detector Regressziós Teszt ==={_RESET}")
    print(f"Gyökér: {ROOT}\n")

    results: list[tuple[str, bool]] = []

    # ── Szekció 1: osztály-szintű tesztek ──
    print(f"{_BOLD}Szekció 1 – Osztály-szintű tesztek (kép nélkül){_RESET}")
    for name, fn in [
        ("ABC enforcement",               test_abc_enforcement),
        ("Kötelező kulcsok (mock)",        test_required_keys_mock),
        ("fit key parítás (mock)",         test_fit_key_parity_mock),
        ("gradient_profile shape",         test_intensity_profile_shape),
        ("_make_empty_fit struktúra",       test_empty_fit_structure),
    ]:
        t0 = time.time()
        try:
            passed = fn()
        except Exception as exc:
            _fail(f"{name}: nem várt kivétel: {exc}")
            passed = False
        results.append((name, passed))
        print(f"          [{time.time()-t0:.2f}s]")

    # ── Szekció 2: valós képek ──
    print(f"\n{_BOLD}Szekció 2 – Valós képek (manifest szükséges){_RESET}")
    entries = _load_sample_entries(n=5)
    if entries:
        print(f"  Betöltve {len(entries)} kép: "
              f"{[e['class'] for e in entries]}")
    else:
        print(f"  {_YELLOW}Manifest nem található – valós képes tesztek kihagyva{_RESET}")

    for name, fn in [
        ("run_v14_pipeline kivétel-mentes", lambda: test_pipeline_runs_without_exception(entries)),
        ("result dict kulcs-parítás",       lambda: test_result_dict_key_parity(entries)),
        ("feature vektor kompatibilitás",   lambda: test_feature_vector_compatible(entries)),
        ("fret_detector_method label",      lambda: test_detector_method_label(entries)),
    ]:
        t0 = time.time()
        try:
            passed = fn()
        except Exception as exc:
            _fail(f"{name}: nem várt kivétel: {exc}")
            traceback.print_exc()
            passed = False
        results.append((name, passed))
        print(f"          [{time.time()-t0:.2f}s]")

    # ── Szekció 3: összehasonlítás ──
    if entries:
        print(f"\n{_BOLD}Szekció 3 – Összehasonlítás (GeometricFretDetector vs. IntensityFretDetector){_RESET}")
        compare_detectors_on_images(entries)

    # ── Összefoglaló ──
    total   = len(results)
    passed  = sum(1 for _, ok in results if ok)
    failed  = total - passed
    color   = _GREEN if failed == 0 else _RED
    print(f"\n{_BOLD}{'=' * 50}{_RESET}")
    print(f"{_BOLD}Összefoglaló: {color}{passed}/{total} teszt PASS{_RESET}")
    if failed:
        print(f"{_RED}Hibás tesztek:{_RESET}")
        for name, ok in results:
            if not ok:
                print(f"  - {name}")
    print(f"{'=' * 50}\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
