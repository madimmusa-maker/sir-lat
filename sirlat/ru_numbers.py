"""(Перенесено из mystery-shorts.) Числа -> русские слова для движков, которые не читают цифры (Silero).

Понимает годы с падежами: «в 1959 году» -> «в тысяча девятьсот пятьдесят девятом году»,
«1959 года», «1990-х», века римскими цифрами: «XX века» -> «двадцатого века».
Остальные числа читаются количественными: «30 лет» -> «тридцать лет».
"""
import re

UNITS = ["ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
UNITS_F = ["ноль", "одна", "две"] + UNITS[3:]
TEENS = ["десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать",
         "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"]
TENS = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят",
        "восемьдесят", "девяносто"]
HUNDREDS = ["", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот",
            "восемьсот", "девятьсот"]

ORD_UNITS = ["", "первый", "второй", "третий", "четвёртый", "пятый", "шестой", "седьмой",
             "восьмой", "девятый"]
ORD_TEENS = ["десятый", "одиннадцатый", "двенадцатый", "тринадцатый", "четырнадцатый",
             "пятнадцатый", "шестнадцатый", "семнадцатый", "восемнадцатый", "девятнадцатый"]
ORD_TENS = ["", "", "двадцатый", "тридцатый", "сороковой", "пятидесятый", "шестидесятый",
            "семидесятый", "восьмидесятый", "девяностый"]
ORD_HUNDREDS = ["", "сотый", "двухсотый", "трёхсотый", "четырёхсотый", "пятисотый",
                "шестисотый", "семисотый", "восьмисотый", "девятисотый"]
GEN_PREFIX = ["", "", "двух", "трёх", "четырёх", "пяти", "шести", "семи", "восьми", "девяти"]


def _below_1000(n, feminine=False):
    words = []
    h, r = divmod(n, 100)
    if h:
        words.append(HUNDREDS[h])
    if 10 <= r < 20:
        words.append(TEENS[r - 10])
    else:
        t, u = divmod(r, 10)
        if t:
            words.append(TENS[t])
        if u:
            words.append((UNITS_F if feminine else UNITS)[u])
    return words


def _thousand_word(n):
    if 10 <= n % 100 < 20 or n % 10 in (0, 5, 6, 7, 8, 9):
        return "тысяч"
    return "тысяча" if n % 10 == 1 else "тысячи"


def cardinal(n):
    if n == 0:
        return "ноль"
    if n >= 1_000_000:
        return " ".join(str(n))  # такие числа в сценариях почти не встречаются
    words = []
    th, rest = divmod(n, 1000)
    if th:
        words += [] if th == 1 else _below_1000(th, feminine=True)
        words.append(_thousand_word(th))
    words += _below_1000(rest)
    return " ".join(words)


def ordinal(n):
    """Порядковое в именительном падеже мужского рода: 1959 -> «...пятьдесят девятый»."""
    if n <= 0 or n >= 1_000_000:
        return cardinal(n)
    if n % 1000 == 0:
        th = n // 1000
        return ("" if th == 1 else GEN_PREFIX[th] if th < 10 else cardinal(th) + " ") + "тысячный"
    r = n % 100
    if n % 100 == 0:
        last_word = ORD_HUNDREDS[(n % 1000) // 100]
        prefix = n - n % 1000
    elif 10 <= r < 20:
        last_word, prefix = ORD_TEENS[r - 10], n - r
    elif r % 10 == 0:
        last_word, prefix = ORD_TENS[r // 10], n - r
    else:
        last_word, prefix = ORD_UNITS[r % 10], n - r % 10
    return (cardinal(prefix) + " " + last_word).strip() if prefix else last_word


def _decline(word_phrase, case):
    """Поменять окончание последнего слова порядкового: case = nom | gen | prep | gen_pl."""
    head, _, last = word_phrase.rpartition(" ")
    if case == "nom":
        new = last
    elif last.endswith("третий"):
        new = last[:-2] + {"gen": "ьего", "prep": "ьем", "gen_pl": "ьих"}[case]
    else:
        new = last[:-2] + {"gen": "ого", "prep": "ом", "gen_pl": "ых"}[case]
    return (head + " " + new).strip()


ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def _roman(s):
    total, prev = 0, 0
    for ch in reversed(s):
        v = ROMAN[ch]
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    return total


def expand(text):
    def year(m):
        num, word = int(m.group(1)), m.group(2)
        lw = word.lower()
        if lw in ("году", "веке"):
            case = "prep"
        elif lw in ("года", "века"):
            case = "gen"
        else:
            case = "nom"
        return _decline(ordinal(num), case) + " " + word

    # «в 1959 году», «1959 года», «XX века»
    text = re.sub(r"\b(\d{1,4})\s+(году|года|год|веке|века|век)\b", year, text, flags=re.I)
    text = re.sub(r"\b([IVXLC]{1,6})\s+(веке|века|век)\b",
                  lambda m: year(re.match(r"(\d+) (\S+)", f"{_roman(m.group(1))} {m.group(2)}")), text)
    # «1990-х», «1990-е»
    text = re.sub(r"\b(\d{1,4})-(?:х|ых)\b", lambda m: _decline(ordinal(int(m.group(1))), "gen_pl"), text)
    text = re.sub(r"\b(\d{1,4})-(?:е|ые)\b",
                  lambda m: _decline(ordinal(int(m.group(1))), "nom")[:-2] + "ые", text)
    # «5-й», «3-м» и т. п. — просто порядковое
    suffix_case = {"й": "nom", "ый": "nom", "ой": "nom", "м": "prep", "ом": "prep", "го": "gen", "ого": "gen"}
    text = re.sub(r"\b(\d{1,4})-(й|ый|ой|м|ом|го|ого)\b",
                  lambda m: _decline(ordinal(int(m.group(1))), suffix_case[m.group(2)]), text)
    # десятичные «2,5» -> «два и пять»
    text = re.sub(r"\b(\d+)[,.](\d+)\b", lambda m: f"{cardinal(int(m.group(1)))} и {cardinal(int(m.group(2)))}", text)
    # всё остальное — количественные (пробелы-разделители тысяч убираем)
    text = re.sub(r"\b(\d{1,3}(?:[  ]\d{3})+)\b", lambda m: re.sub(r"\D", "", m.group(1)), text)
    text = re.sub(r"\d+", lambda m: cardinal(int(m.group(0))), text)
    return text
