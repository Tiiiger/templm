import calendar
import re

_e2e_field_transformations_complex = {
    ("food", "Fast food"): ["Fast food", "fast food"],
    ("customer rating", "1 out of 5"): ["1 out of 5", "low"],
    ("customer rating", "3 out of 5"): ["3 out of 5", "average"],
    ("customer rating", "5 out of 5"): ["5 out of 5", "high"],
    ("priceRange", "less than £20"): ["less than £20", "cheap"],
    ("priceRange", "£20-25"): ["£20-25", "moderate"],
    ("priceRange", "more than £30"): ["more than £30", "high"],
    ("familyFriendly", "yes"): [
        "family friendly",
        "kid friendly",
        "children friendly",
        "family-friendly",
        "child friendly",
    ],
    ("familyFriendly", "no"): [
        "not family friendly",
        "not kid friendly",
        "not children friendly",
        "not family-friendly",
        "not child friendly",
    ],
}
_e2e_field_transformations_simple = {
    ("food", "Fast food"): ["Fast food", "fast food"],
    ("familyFriendly", "yes"): ["family friendly", "family-friendly",],
    ("familyFriendly", "no"): ["not family friendly", "not family-friendly",],
}


def e2e_field_transformation(data_table, complex=True):
    if complex:
        _e2e_transform = _e2e_field_transformations_complex
    else:
        _e2e_transform = _e2e_field_transformations_simple

    new_data_table = dict()
    for field_name, field_content in data_table.items():
        if (field_name, field_content) in _e2e_transform:
            field_content = _e2e_transform[(field_name, field_content)]
        else:
            field_content = [field_content]
        new_data_table[field_name] = field_content
    new_data_table["article"] = ["a", "an"]

    return new_data_table


class SearchDate:
    regex_exps = [
        re.compile(f"^(\d{{1,2}})[,\s]+([a-zA-Z]+)[,\s]+(\d+)$"),  # day month year
        re.compile(f"^([a-zA-Z]+)[,\s]+(\d{{3,4}})$"),  # month year
        re.compile(f"^([a-zA-Z]+)[,\s]+(\d{{1,2}})[,\s]+(\d+)$"),  # month day year
        re.compile(f"^(\d+)[,\s]+([a-zA-Z]+)[,\s]+(\d{{1,2}})$"),  # year month day
        re.compile(f"^(\d{{3,4}})\s+(\d{{1,2}})\s+(\d{{1,2}})$"),  # year month day
        re.compile(f"^(\d{{1,2}})\s+(\d{{1,2}})\s+(\d{{3,4}})$"),  # month day year
        re.compile(f"^(\d{{3,4}})$"),  # year
        re.compile(f"^(\d{{3,4}})\s+(\d{{1,2}})$"),
    ]  # year month

    regex_date_assignments = [
        [3, 2, 1],  # index into year, month, day pattern
        [2, 1, 0],
        [3, 1, 2],
        [1, 2, 3],
        [1, 2, 3],
        [3, 1, 2],
        [1, 0, 0],
        [1, 2, 0],
    ]  # 0 means absent

    def __init__(self):

        self.year, self.month, self.day = 0, 0, 0
        self.target_date_strings = set()

    def find_date_in_input(self, input_string):

        input_string = input_string.replace("yes", "")
        input_string = input_string.replace("ca", "")
        input_string = input_string.replace("a.d", "")
        input_string = input_string.replace("c.", "")
        input_string = input_string.replace("?", "")
        input_string = input_string.replace("-", " ")
        input_string = input_string.replace(".", "")
        input_string = input_string.replace("around", "")
        input_string = re.sub(r"\(.*\)", "", input_string)
        input_string = input_string.strip()

        for idx, month_abre in enumerate(list(calendar.month_abbr)):
            input_string = input_string.replace(
                f"{month_abre.lower()} ", f"{calendar.month_name[idx].lower()} "
            )

        found, found_idx, search_ojb = False, -1, None
        for idx, exp in enumerate(self.regex_exps):
            search_ojb = re.search(exp, input_string)
            if search_ojb is not None:
                found, found_idx = True, idx
                break

        if found:
            self.year, self.month, self.day = [
                search_ojb.group(item) if item != 0 else "0"
                for item in self.regex_date_assignments[found_idx]
            ]
            if not self.month.isnumeric():
                self.month = (
                    list(calendar.month_name).index(self.month.capitalize())
                    if self.month.capitalize() in list(calendar.month_name)
                    else None
                )
            if self.month is None:
                return False
            self.year, self.month, self.day = (
                int(self.year),
                int(self.month),
                int(self.day),
            )

        if self.month > 12:
            self.month, self.day = self.day, self.month

        return found

    def generate_equiv(self, input_string):

        date_strings = self.target_date_strings
        date_strings.add(input_string)
        found = self.find_date_in_input(input_string)
        day_string, month_string, year_string = "", "", ""

        if found:
            if self.day != 0:
                day_string = {f"{self.day}", f"{self.day:02}"}
            if self.month != 0:
                month_string = {
                    f"{self.month}",
                    f"{self.month:02}",
                    f"{calendar.month_abbr[self.month].lower()}",
                    f"{calendar.month_name[self.month].lower()}",
                }
            if self.year != 0:
                year_string = [f"{self.year}"]
        return day_string, month_string, year_string


def sb_field_transformation(input_table):
    data_table = input_table.copy()

    if "birth_date" in data_table:
        search_date = SearchDate()
        day_str, mon_str, year_str = search_date.generate_equiv(
            data_table["birth_date"]
        )
        if day_str != "":
            data_table["birth_date_day"] = day_str
        if mon_str != "":
            data_table["birth_date_month"] = mon_str
        if year_str != "":
            data_table["birth_date_year"] = year_str
        del data_table["birth_date"]

    if "death_date" in data_table:
        search_date = SearchDate()
        day_str, mon_str, year_str = search_date.generate_equiv(
            data_table["death_date"]
        )
        if day_str != "":
            data_table["death_date_day"] = day_str
        if mon_str != "":
            data_table["death_date_month"] = mon_str
        if year_str != "":
            data_table["death_date_year"] = year_str
        del data_table["death_date"]

    if "gender" in data_table:
        gender = data_table["gender"]
        if gender == "male":
            data_table["relation"] = ["son"]
            data_table["pronoun_a"] = ["he"]
            data_table["pronoun_b"] = ["him"]
            data_table["pronoun_c"] = ["his"]
            data_table["pronoun_d"] = ["man"]
        if gender == "female":
            data_table["relation"] = ["daughter"]
            data_table["pronoun_a"] = ["she"]
            data_table["pronoun_b"] = ["her"]
            data_table["pronoun_c"] = ["her"]
            data_table["pronoun_d"] = ["woman"]
        if gender == "non-binary":
            data_table["pronoun_a"] = ["they"]
            data_table["pronoun_b"] = ["them"]
            data_table["pronoun_c"] = ["their"]

    to_delete = []
    for k, v in data_table.items():
        if v in ["none", "", "no info", "unknown"]:
            to_delete.append(k)
            continue
        if isinstance(v, str):
            if v[-1] in [",", "."]:
                v = v[:-1]
            data_table[k] = [v]

    for k in to_delete:
        del data_table[k]

    data_table["article"] = ["a", "an"]
    data_table["be"] = ["is", "are", "was", "were"]
    data_table["number"] = [
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
    ]

    del data_table["notable_type"]

    return data_table
