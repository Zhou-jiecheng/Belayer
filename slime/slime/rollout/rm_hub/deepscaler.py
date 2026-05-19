import logging
import os
import time

from .math_utils import extract_answer, grade_answer_mathd, grade_answer_sympy

logger = logging.getLogger(__name__)
SLOW_RM_LOG_SEC = float(os.getenv("SLIME_RM_SLOW_LOG_SEC", "1.0"))


def get_deepscaler_rule_based_reward(response, label):
    start_time = time.monotonic()
    if "</think>" in response:
        model_solution = response.split("</think>")[-1]
    elif "###Response" in response:
        model_solution = response.split("###Response")[1]
    else:
        return 0

    model_answer = extract_answer(model_solution)
    if model_answer is None:
        return 0
    if label == "":
        return 0

    # Convert single answer to list for uniform processing
    assert isinstance(label, (str, float, int))
    ground_truths = [label]

    # Process each ground truth
    processed_ground_truths = []
    for truth in ground_truths:
        truth = str(truth)
        if "\\boxed" in truth:
            processed_truth = extract_answer(truth)
            if processed_truth is not None:
                processed_ground_truths.append(processed_truth)
        else:
            processed_ground_truths.append(truth)

    if not processed_ground_truths:
        return 0

    # Check against all possible correct answers
    for ground_truth in processed_ground_truths:
        check_start_time = time.monotonic()
        mathd_correct = grade_answer_mathd(model_answer, ground_truth)
        mathd_elapsed = time.monotonic() - check_start_time
        if mathd_elapsed >= SLOW_RM_LOG_SEC:
            logger.warning(
                "deepscaler RM slow in grade_answer_mathd: elapsed=%.2fs answer_len=%s gt_len=%s",
                mathd_elapsed,
                len(model_answer),
                len(str(ground_truth)),
            )

        sympy_correct = False
        if not mathd_correct:
            sympy_start_time = time.monotonic()
            sympy_correct = grade_answer_sympy(model_answer, ground_truth)
            sympy_elapsed = time.monotonic() - sympy_start_time
            if sympy_elapsed >= SLOW_RM_LOG_SEC:
                logger.warning(
                    "deepscaler RM slow in grade_answer_sympy: elapsed=%.2fs answer_len=%s gt_len=%s answer_tail=%r gt=%r",
                    sympy_elapsed,
                    len(model_answer),
                    len(str(ground_truth)),
                    model_answer[-120:],
                    str(ground_truth)[:120],
                )

        is_correct = mathd_correct or sympy_correct
        if is_correct:
            elapsed = time.monotonic() - start_time
            if elapsed >= SLOW_RM_LOG_SEC:
                logger.warning(
                    "deepscaler RM total slow: elapsed=%.2fs matched=True answer_len=%s ground_truths=%s",
                    elapsed,
                    len(model_answer),
                    len(processed_ground_truths),
                )
            return 1

    elapsed = time.monotonic() - start_time
    if elapsed >= SLOW_RM_LOG_SEC:
        logger.warning(
            "deepscaler RM total slow: elapsed=%.2fs matched=False answer_len=%s ground_truths=%s",
            elapsed,
            len(model_answer),
            len(processed_ground_truths),
        )

    return 0
