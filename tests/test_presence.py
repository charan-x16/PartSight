"""CAD-only presence decisions (Problem Statement 1) on a small synthetic scene."""

import cv2
import numpy as np
import pytest

from src.data_inspection import PartMapping
from src.detector import AlignmentResult, extract_part_regions
from src.presence import classify_presence, part_type

HEIGHT, WIDTH = 400, 640
NUTS = {1: (120, 120), 2: (280, 120), 3: (440, 120), 4: (120, 290)}  # value -> centre
BOX = (5, (300, 230, 560, 360))  # value, rectangle
TINY = (6, (590, 30))


def _hexagon(center, radius=45):
    angles = np.deg2rad(np.arange(0, 360, 60) + 15)
    return np.array([[center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)] for a in angles], np.int32)


@pytest.fixture(scope="module")
def scene():
    rng = np.random.default_rng(3)
    ids = np.zeros((HEIGHT, WIDTH), np.uint8)
    cad = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    photo = np.full((HEIGHT, WIDTH, 3), 100, np.uint8)  # flat plate
    texture = np.clip(150 + rng.normal(0, 60, (HEIGHT, WIDTH)), 0, 255).astype(np.uint8)
    for value, center in NUTS.items():
        hexagon = _hexagon(center)
        cv2.fillPoly(ids, [hexagon], value)
        cv2.fillPoly(cad, [hexagon], (200, 180, 40))
        cv2.polylines(cad, [hexagon], True, (60, 50, 10), 3)
        inside = np.zeros((HEIGHT, WIDTH), np.uint8)
        cv2.fillPoly(inside, [hexagon], 1)
        if value in (1, 2):  # intact parts
            photo[inside > 0] = texture[inside > 0, None]
            cv2.polylines(photo, [hexagon], True, (20, 20, 20), 3)
        elif value == 4:  # covered by a smooth, low-contrast patch: not blank, but unlike its siblings
            ramp = np.tile(np.linspace(94, 108, WIDTH), (HEIGHT, 1)).astype(np.uint8)
            photo[inside > 0] = ramp[inside > 0, None]
        # value 3 stays plate-coloured: painted out
    x0, y0, x1, y1 = BOX[1]
    ids[y0:y1, x0:x1] = BOX[0]
    cv2.rectangle(cad, (x0, y0), (x1 - 1, y1 - 1), (40, 90, 160), -1)
    cv2.rectangle(photo, (x0, y0), (x1 - 1, y1 - 1), (170, 170, 170), -1)
    ids[TINY[1][1]:TINY[1][1] + 5, TINY[1][0]:TINY[1][0] + 5] = TINY[0]
    mapping = [PartMapping(f"hex_nut_{value}", value) for value in NUTS] + [
        PartMapping("box", BOX[0]), PartMapping("tiny_pin_1", TINY[0]),
    ]
    regions = extract_part_regions(ids, mapping)
    records = classify_presence(regions, photo, cad, AlignmentResult(True, np.eye(3), "ok"))
    return {record["pixel_value"]: record for record in records}


def test_intact_parts_are_present(scene):
    assert scene[1]["status"] == "PRESENT" and scene[2]["status"] == "PRESENT"


def test_blank_region_is_absent(scene):
    assert scene[3]["status"] == "MISSING"
    assert "blank" in scene[3]["diagnostic"]


def test_part_much_weaker_than_its_siblings_is_absent(scene):
    record = scene[4]
    assert record["status"] == "MISSING", record["diagnostic"]
    assert record["relative_evidence"] < 0.6
    assert record["sibling_count"] == 2  # the blank sibling is not used as a reference


def test_part_without_siblings_uses_its_outline_support(scene):
    record = scene[BOX[0]]
    assert record["status"] == "PRESENT", record["diagnostic"]
    assert "No same-type part" in record["diagnostic"]


def test_part_too_small_in_the_cad_view_is_not_guessed(scene):
    assert scene[TINY[0]]["status"] == "NOT_VISIBLE"


def test_alignment_failure_gives_uncertain_for_every_part():
    regions = extract_part_regions(np.array([[0, 1]], np.uint8), [PartMapping("part", 1)])
    records = classify_presence(regions, None, np.zeros((1, 2, 3), np.uint8), AlignmentResult(False, None, "no match"))
    assert records[0]["status"] == "UNCERTAIN" and "no match" in records[0]["diagnostic"]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("large_washer_5", "large_washer"),
        ("hex nut style 1 gradeab_iso-7_BRep_1.019", "hex nut style 1 gradeab_iso-7_BRep"),
        ("Large_washer-2", "Large_washer"),
        ("Bolts_M8_4", "Bolts_M8"),
        ("box", "box"),
    ],
)
def test_part_type_strips_the_instance_number(name, expected):
    assert part_type(name) == expected
