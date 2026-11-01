"""Prompt formatting for standard and reasoning-augmented experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum

from clinical_re_bilingual.prompting.instructions import task_instruction
from clinical_re_bilingual.prompting.labels import relation_name
from clinical_re_bilingual.schema import Language, RelationExample, RelationLabel


class PromptStrategy(str, Enum):
    STANDARD = "standard"
    ZS_STATIC_COT = "zs_static_cot"
    FS_STATIC_COT = "fs_static_cot"
    SELF_QUESTIONING = "self_questioning"
    GOLD_LABEL_COT = "gold_label_cot"
    OUTPUT_FORMAT_COT = "output_format_cot"


_RELATION_VERBS = {
    RelationLabel.TrIP: "IMPROVES",
    RelationLabel.TrWP: "WORSENS",
    RelationLabel.TrCP: "CAUSES",
    RelationLabel.TrAP: "IS ADMINISTERED FOR",
    RelationLabel.TrNAP: "IS NOT ADMINISTERED BECAUSE OF",
    RelationLabel.PIP: "INDICATES",
    RelationLabel.TeRP: "REVEALS",
    RelationLabel.TeCP: "IS CONDUCTED TO INVESTIGATE",
}


_ZERO_SHOT_COT = {
    Language.ENGLISH: (
        "Let's think step by step. First identify whether each entity is a treatment, "
        "test, or medical problem. Then determine how the two entities are connected in "
        "the sentence: a treatment may improve, worsen, cause, be administered for, or "
        "not be administered because of a medical problem; a test may reveal or be "
        "conducted to investigate a medical problem; or one medical problem may indicate "
        "another. Conclude with the single appropriate relation type."
    ),
    Language.TURKISH: (
        "Adım adım düşünelim. Önce her varlığın tedavi, test veya medikal problem olup "
        "olmadığını belirle. Ardından cümledeki bağlantıyı değerlendir: bir tedavi medikal "
        "problemi iyileştirebilir, kötüleştirebilir, probleme neden olabilir, problem için "
        "uygulanabilir veya problem nedeniyle uygulanmayabilir; bir test problemi ortaya "
        "çıkarabilir veya araştırmak için uygulanabilir; ya da bir medikal problem başka "
        "bir medikal problemi gösterebilir. Tek bir uygun ilişki türüyle sonuçlandır."
    ),
}


def _question(example: RelationExample) -> str:
    if example.language is Language.ENGLISH:
        return (
            f"Context: {example.sentence}\nGiven the context, what is the relation between "
            f'"{example.subject}" and "{example.object}"?'
        )
    return (
        f'Bağlam: {example.sentence}\nBağlam göz önüne alındığında, "{example.subject}" '
        f'ile "{example.object}" arasındaki ilişki nedir?'
    )


def _structured_explanation(example: RelationExample) -> str:
    if example.language is Language.ENGLISH:
        return (
            f"Because {example.subject_type} [{example.subject}] "
            f"{_RELATION_VERBS[example.label]} "
            f"{example.object_type} [{example.object}]."
        )
    return (
        f"Çünkü {example.subject_type} [{example.subject}] ile "
        f"{example.object_type} [{example.object}] arasındaki ilişki "
        f"{relation_name(example.label, example.language)} şeklindedir."
    )


class PromptBuilder:
    def __init__(self, language: Language, strategy: PromptStrategy) -> None:
        self.language = language
        self.strategy = strategy

    def build(
        self,
        query: RelationExample,
        demonstrations: Sequence[RelationExample],
        *,
        rationales: Mapping[str, str] | None = None,
    ) -> str:
        if query.language is not self.language:
            raise ValueError("query language does not match prompt language")
        if any(item.language is not self.language for item in demonstrations):
            raise ValueError("demonstration language does not match prompt language")
        if len({item.example_id for item in demonstrations}) != len(demonstrations):
            raise ValueError("demonstrations must not contain duplicate IDs")
        sections = [task_instruction(self.language)]
        if self.strategy is PromptStrategy.ZS_STATIC_COT:
            if demonstrations:
                raise ValueError("zero-shot static CoT does not accept demonstrations")
            sections.append(_ZERO_SHOT_COT[self.language])
        elif not demonstrations:
            raise ValueError(f"strategy {self.strategy.value} requires demonstrations")
        for example in demonstrations:
            response = relation_name(example.label, self.language)
            if self.strategy is PromptStrategy.OUTPUT_FORMAT_COT:
                response = f"{response}. {_structured_explanation(example)}"
            elif self.strategy in {
                PromptStrategy.FS_STATIC_COT,
                PromptStrategy.SELF_QUESTIONING,
                PromptStrategy.GOLD_LABEL_COT,
            }:
                if rationales is None or not rationales.get(example.example_id, "").strip():
                    raise ValueError(
                        f"strategy {self.strategy.value} requires a rationale for "
                        f"{example.example_id}"
                    )
                response = f"{response}. {rationales[example.example_id].strip()}"
            answer_word = "Response" if self.language is Language.ENGLISH else "Cevap"
            sections.append(f"{_question(example)}\n{answer_word}: {response}")
        answer_word = "Response" if self.language is Language.ENGLISH else "Cevap"
        sections.append(f"{_question(query)}\n{answer_word}:")
        return "\n\n".join(sections)
