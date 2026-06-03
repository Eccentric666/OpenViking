"""QueryNormalizer — low-cost text normalization for stable template matching.

Design principle: do not classify, do not decide backend, only standardize text.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Lightweight time expressions to preserve during normalization
_TIME_PATTERNS = re.compile(
    r"(上次|最近|昨天|今天|明天|上周|下周|上个月|下个月|"
    r"去年|今年|明年|刚刚|刚才|之前|以后|"
    r"\d{4}年|\d{1,2}月|\d{1,2}日|"
    r"January|February|March|April|May|June|July|August|September|October|November|December|"
    r"last\s+(week|month|year)|next\s+(week|month|year)|"
    r"yesterday|today|tomorrow)",
    re.IGNORECASE,
)

# Punctuation and whitespace to collapse (CJK + ASCII)
_PUNCTUATION_COLLAPSE = re.compile(r"[，。！？、；：\"'（）【】《》,.!?;:'\"()\[\]<>\s]+")

# VikingBot prompt prefix patterns — these prefixes degrade embedding quality
# when passed through to MemRouter template matching.
_VIKINGBOT_PREFIX_PATTERNS = [
    # Full prefix: "Current date: YYYY-MM-DD. Before answering, search... Then answer..."
    re.compile(
        r"^Current date:\s*\d{4}-\d{2}-\d{2}\.\s*"
        r"Before answering, search the user's OpenViking memory using the "
        r"memory search tool exactly once\.\s*"
        r"Then answer the question directly:\s*",
        re.IGNORECASE,
    ),
    # Date + direct answer (no search instruction)
    re.compile(
        r"^Current date:\s*\d{4}-\d{2}-\d{2}\.\s*"
        r"Answer the question directly:\s*",
        re.IGNORECASE,
    ),
    # Search instruction only (no date)
    re.compile(
        r"^Before answering, search the user's OpenViking memory using the "
        r"memory search tool exactly once\.\s*"
        r"Then answer the question directly:\s*",
        re.IGNORECASE,
    ),
    # Direct answer only
    re.compile(
        r"^Answer the question directly:\s*",
        re.IGNORECASE,
    ),
]

# Common person names in the LoCoMo dataset + English first names.
# These are replaced with a generic "person" token so hard-negative
# examples written with PERSON_A placeholders actually match.
_PERSON_NAMES = [
    "caroline", "melanie", "jon", "john", "maria", "gina", "sam",
    "nate", "joanna", "dave", "calvin", "audrey", "evan", "james",
    "rachel", "emily", "lisa", "tom", "michael", "david", "sarah",
    "jessica", "jennifer", "matthew", "daniel", "christopher", "andrew",
    "joshua", "nicholas", "ryan", "brandon", "justin", "benjamin",
    "samuel", "jonathan", "joseph", "william", "robert", "richard",
    "thomas", "charles", "anthony", "mark", "donald", "steven", "paul",
    "kenneth", "kevin", "brian", "george", "timothy", "ronald", "edward",
    "jason", "jeffrey", "jacob", "gary", "eric", "stephen", "larry",
    "scott", "frank", "gregory", "raymond", "alexander", "patrick",
    "jack", "dennis", "jerry", "tyler", "aaron", "jose", "adam",
    "nathan", "henry", "zachary", "douglas", "peter", "kyle", "walter",
    "ethan", "jeremy", "harold", "keith", "christian", "roger", "noah",
    "gerald", "terry", "sean", "austin", "carl", "arthur", "lawrence",
    "dylan", "jesse", "jordan", "bryan", "billy", "bruce", "gabriel",
    "joe", "logan", "alan", "juan", "wayne", "elijah", "randy", "roy",
    "vincent", "ralph", "eugene", "russell", "bobby", "mason", "philip",
    "louis", "mary", "patricia", "linda", "elizabeth", "barbara", "susan",
    "karen", "nancy", "betty", "margaret", "sandra", "ashley", "kimberly",
    "donna", "michelle", "dorothy", "carol", "amanda", "melissa", "deborah",
    "stephanie", "rebecca", "laura", "sharon", "cynthia", "kathleen", "amy",
    "shirley", "anna", "brenda", "pamela", "emma", "nicole", "helen",
    "samantha", "katherine", "christine", "debra", "catherine", "carolyn",
    "janet", "ruth", "heather", "diane", "virginia", "julie", "joyce",
    "victoria", "olivia", "kelly", "christina", "lauren", "joan", "evelyn",
    "judith", "megan", "cheryl", "andrea", "hannah", "martha", "jacqueline",
    "frances", "gloria", "ann", "teresa", "kathryn", "sara", "janice",
    "jean", "alice", "madison", "doris", "abigail", "julia", "judy",
    "grace", "amber", "denise", "marilyn", "danielle", "beverly", "isabella",
    "theresa", "sophia", "marie", "diana", "brittany", "natalie", "charlotte",
    "amelia", "harper", "avery", "sofia", "camila", "aria", "scarlett",
    "luna", "chloe", "penelope", "layla", "riley", "zoey", "nora", "lily",
    "eleanor", "lillian", "addison", "aubrey", "ellie", "stella", "zoe",
    "leah", "hazel", "violet", "aurora", "savannah", "brooklyn", "bella",
    "claire", "skylar", "lucy", "paisley", "everly", "nova", "genesis",
    "emilia", "kennedy", "maya", "willow", "kinsley", "naomi", "elena",
    "allison", "gabriella", "alice", "madelyn", "cora", "ruby", "eva",
    "serenity", "autumn", "adeline", "hailey", "gianna", "valentina",
    "isla", "eliana", "quinn", "nevaeh", "ivy", "sadie", "piper", "lydia",
    "alexa", "josephine", "emery", "delilah", "arianna", "vivian", "kaylee",
    "sophie", "brielle", "madeline",
]
_PERSON_NAME_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(n) for n in _PERSON_NAMES) + r")\b",
    re.IGNORECASE,
)

# Template placeholder tokens (PERSON_A, PERSON_B, ITEM_X, etc.)
_PLACEHOLDER_PATTERN = re.compile(
    r"\b(PERSON_[A-Z]|ITEM_[A-Z]+|EVENT_[A-Z]+|PLACE_[A-Z]+|ACTION_[A-Z]+|ACTIVITY_[A-Z]+|DATE_[A-Z]+|MONTH_[A-Z]+|YEAR_[A-Z]+|TIME_PERIOD_[A-Z]+|NOUN_[A-Z]+|HOBBY_[A-Z]+|JOB_[A-Z]+|GROUP_[A-Z]+|ORGANIZATION_[A-Z]+|BAND_[A-Z]+|MOVIE_[A-Z]+|BOOK_TITLE|COMPETITION_[A-Z]+|RELATIVE_[A-Z]+|CREATIVE_WORK|CARE_OF_ITEM|HOLIDAY_[A-Z]+|ACCIDENT_TYPE|SKILL_[A-Z]+|PROJECT_[A-Z]+|FOOD_[A-Z]+|PET_[A-Z]+|TRIP_[A-Z]+)\b",
    re.IGNORECASE,
)


def _replace_placeholder(match: "re.Match[str]") -> str:
    """Map template placeholder tokens to semantically meaningful generic words.

    Previously all placeholders were replaced with ``"person"``, which destroyed
    the semantic distinction between actions, places, items, etc.  Now each
    placeholder family maps to a domain-appropriate token so that embedding
    similarity reflects the *structural role* of the slot, not just its presence.
    """
    token = match.group(1).upper()
    if token.startswith("PERSON"):
        return "person"
    if token.startswith("ACTION") or token.startswith("ACTIVITY"):
        return "action"
    if token.startswith("ITEM"):
        return "item"
    if token.startswith("PLACE"):
        return "place"
    if token.startswith("EVENT"):
        return "event"
    if token.startswith("JOB"):
        return "job"
    if (
        token.startswith("DATE")
        or token.startswith("MONTH")
        or token.startswith("YEAR")
        or token.startswith("TIME_PERIOD")
    ):
        return "date"
    if token.startswith("GROUP") or token.startswith("ORGANIZATION"):
        return "group"
    if token.startswith("BAND"):
        return "band"
    if token.startswith("MOVIE"):
        return "movie"
    if token == "BOOK_TITLE":
        return "book"
    if token.startswith("HOBBY"):
        return "hobby"
    if token.startswith("SKILL"):
        return "skill"
    if token.startswith("PROJECT"):
        return "project"
    if token.startswith("FOOD"):
        return "food"
    if token.startswith("PET"):
        return "pet"
    if token.startswith("TRIP"):
        return "trip"
    if token.startswith("NOUN"):
        return "thing"
    if token.startswith("COMPETITION"):
        return "competition"
    if token.startswith("RELATIVE"):
        return "relative"
    if token == "CREATIVE_WORK":
        return "work"
    if token == "CARE_OF_ITEM":
        return "care"
    if token.startswith("HOLIDAY"):
        return "holiday"
    if token.startswith("ACCIDENT"):
        return "accident"
    return "thing"


class QueryNormalizer:
    """Normalize raw user queries into stable matching text."""

    def normalize(self, raw_query: str) -> str:
        """Normalize a raw user query.

        Steps:
            1. Strip leading/trailing whitespace.
            2. Replace common punctuation with single spaces.
            3. Collapse consecutive whitespace.
            4. Lowercase (English only, preserves CJK).
            5. Replace person names with "person" token.
            6. Replace template placeholders with "person" / "item" / "event" tokens.

        Args:
            raw_query: The original user query.

        Returns:
            Normalized query string suitable for embedding and template matching.
        """
        if not raw_query:
            logger.debug("normalize received empty query")
            return ""

        # Step 1: strip
        text = raw_query.strip()

        # Step 1b: strip VikingBot prompt prefix if present
        # These prefixes are injected by VikingBot/E2E eval scripts and
        # severely degrade embedding-based template matching scores.
        for pattern in _VIKINGBOT_PREFIX_PATTERNS:
            match = pattern.search(text)
            if match:
                text = text[match.end():].strip()
                logger.debug("Stripped VikingBot prefix. Remaining: %s", text[:80])
                break

        # Step 2: replace punctuation clusters with single space
        text = _PUNCTUATION_COLLAPSE.sub(" ", text)

        # Step 3: collapse whitespace
        text = " ".join(text.split())

        # Step 4: lowercase (preserves CJK characters safely)
        text = text.lower()

        # Step 5: replace known person names with "person"
        text = _PERSON_NAME_PATTERN.sub("person", text)

        # Step 6: replace template placeholders with generic tokens
        text = _PLACEHOLDER_PATTERN.sub(_replace_placeholder, text)

        logger.debug(
            "Normalized query: raw='%s' -> normalized='%s'",
            raw_query,
            text,
        )
        return text