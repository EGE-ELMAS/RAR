"""Task instructions shared across model providers."""

from clinical_re_bilingual.prompting.labels import relation_code, relation_name
from clinical_re_bilingual.schema import Language, RelationLabel


def task_instruction(language: Language) -> str:
    relations = "\n".join(
        f"{relation_code(label, language)}: {relation_name(label, language)}"
        for label in RelationLabel
    )
    if language is Language.ENGLISH:
        opening = (
            "You are a knowledgeable AI assistant tasked with extracting the relationship "
            "between two entities in a given sentence. Classify the relationship as exactly "
            "one of the following relation types."
        )
        closing = "Return only the relation code or its exact relation name."
    else:
        opening = (
            "Bir cümledeki iki varlık arasındaki ilişkiyi çıkarmakla görevli bilgili bir "
            "yapay zekâ asistanısın. İlişkiyi aşağıdaki türlerden tam olarak biri olarak "
            "sınıflandır."
        )
        closing = "Yalnızca ilişki kodunu veya tam ilişki adını döndür."
    return f"{opening}\n\n{relations}\n\n{closing}"
