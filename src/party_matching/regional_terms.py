"""Conservative, organization-agnostic vocabulary for regional name variants.

This is text interpretation, not evidence of corporate ownership. A location
term is useful only after the matcher has found a unique verified name core.
The list is deliberately explicit and reviewable; unknown terms cause this
optional route to abstain rather than being guessed from a model.
"""

from __future__ import annotations

from .domain import normalize_name


_LOCATIONS = """
africa|asia|europe|oceania|north america|south america|latin america|central america|
united states|united states of america|usa|us|america|canada|mexico|brazil|brasil|
argentina|chile|colombia|peru|uruguay|paraguay|ecuador|bolivia|venezuela|
united kingdom|uk|britain|england|scotland|wales|ireland|france|germany|deutschland|
spain|portugal|italy|netherlands|belgium|luxembourg|switzerland|austria|poland|
czech republic|czechia|slovakia|hungary|romania|bulgaria|greece|turkey|sweden|
norway|denmark|finland|iceland|estonia|latvia|lithuania|ukraine|croatia|serbia|
slovenia|bosnia|albania|malta|cyprus|russia|israel|saudi arabia|united arab emirates|
uae|qatar|kuwait|bahrain|oman|egypt|morocco|nigeria|kenya|ghana|south africa|
india|pakistan|bangladesh|sri lanka|nepal|china|japan|south korea|korea|taiwan|
hong kong|singapore|malaysia|indonesia|thailand|vietnam|philippines|australia|
new zealand|new york|los angeles|san francisco|chicago|houston|dallas|boston|
seattle|atlanta|miami|toronto|vancouver|montreal|london|paris|berlin|munich|
amsterdam|madrid|rome|milan|tokyo|osaka|seoul|beijing|shanghai|mumbai|delhi|
bangalore|bengaluru|sao paulo|rio de janeiro|dubai|sydney|melbourne|
alabama|alaska|arizona|arkansas|california|colorado|connecticut|delaware|florida|
georgia|hawaii|idaho|illinois|indiana|iowa|kansas|kentucky|louisiana|maine|
maryland|massachusetts|michigan|minnesota|mississippi|missouri|montana|nebraska|
nevada|new hampshire|new jersey|new mexico|north carolina|north dakota|ohio|
oklahoma|oregon|pennsylvania|rhode island|south carolina|south dakota|tennessee|
texas|utah|vermont|virginia|washington|west virginia|wisconsin|wyoming|
north|south|east|west|northern|southern|eastern|western|central|
latam|emea|apac|dach|anz
"""

# Jurisdictional adjectives are allowed only as extra text, never as proof of
# identity. These are common forms, not an exhaustive language dictionary.
_ADJECTIVES = """
american|canadian|mexican|brazilian|british|english|irish|french|german|spanish|
portuguese|italian|dutch|belgian|swiss|austrian|polish|swedish|norwegian|danish|
finnish|turkish|israeli|emirati|egyptian|nigerian|kenyan|south african|indian|
pakistani|chinese|japanese|korean|taiwanese|singaporean|malaysian|indonesian|
thai|vietnamese|filipino|australian|zealand
"""

# Do not include ownership-signalling words such as subsidiary or holdings:
# they can denote a different legal entity, even when a parent brand appears.
_STRUCTURE = """
division|branch|office|region|regional|territory|operations|operation|department|unit|
team|district|area|market
"""

_LEGAL = """
inc|incorporated|corp|corporation|company|co|llc|llp|ltd|limited|plc|gmbh|
ag|bv|nv|sa|sas|sarl|spa|pty|pte|pvt|srl|ltda|sro
"""


def _phrases(source: str) -> set[tuple[str, ...]]:
    return {tuple(normalize_name(value).split()) for value in source.split("|")
            if normalize_name(value)}


LOCATION_PHRASES = _phrases(_LOCATIONS) | _phrases(_ADJECTIVES)
STRUCTURE_PHRASES = _phrases(_STRUCTURE)
LEGAL_PHRASES = _phrases(_LEGAL)
IGNORED_LEGAL_TOKENS = {token for phrase in LEGAL_PHRASES for token in phrase if len(phrase) == 1}
ALLOWED_PHRASES = sorted(LOCATION_PHRASES | STRUCTURE_PHRASES | LEGAL_PHRASES,
                         key=lambda phrase: -len(phrase))


def explain_region_tokens(tokens: list[str]) -> tuple[bool, bool]:
    """Return (all explained, has location) using longest phrase matches."""
    offset = 0
    has_location = False
    while offset < len(tokens):
        phrase = next((entry for entry in ALLOWED_PHRASES
                       if tuple(tokens[offset:offset + len(entry)]) == entry), None)
        if phrase is None:
            return False, has_location
        has_location |= phrase in LOCATION_PHRASES
        offset += len(phrase)
    return True, has_location
