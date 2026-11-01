"""Templates used to generate reasoning for retrieved demonstrations."""

from clinical_re_bilingual.prompting.builder import PromptStrategy
from clinical_re_bilingual.prompting.labels import relation_name
from clinical_re_bilingual.schema import Language, RelationExample


def rationale_request(example: RelationExample, strategy: PromptStrategy) -> str:
    relation = relation_name(example.label, example.language)
    if strategy is PromptStrategy.GOLD_LABEL_COT:
        if example.language is Language.ENGLISH:
            return (
                f'What clues make the relation between "{example.subject}" and '
                f'"{example.object}" {relation} in the sentence "{example.sentence}"? '
                "Answer in at most 100 words."
            )
        return (
            f'"{example.sentence}" cümlesinde "{example.subject}" ile '
            f'"{example.object}" arasındaki ilişkinin {relation} olmasını gösteren '
            "ipuçları nelerdir? En fazla 100 kelimeyle cevapla."
        )
    if strategy is PromptStrategy.SELF_QUESTIONING:
        if example.language is Language.ENGLISH:
            return (
                f"Context: {example.sentence}\nIdentify the relation between "
                f'"{example.subject}" and "{example.object}". Generate and briefly answer '
                "diagnostic questions considering TrIP, TrWP, TrCP, TrAP, TrNAP, PIP, "
                f"TeRP, and TeCP, then conclude that the relation is {relation}."
            )
        return (
            f'Bağlam: {example.sentence}\n"{example.subject}" ile "{example.object}" '
            "arasındaki ilişkiyi belirlemek için tanısal sorular üretip kısaca yanıtla; "
            f"ardından ilişkinin {relation} olduğu sonucuna var."
        )
    raise ValueError(f"strategy {strategy.value} does not generate dynamic rationales")
