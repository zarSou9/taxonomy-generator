from pydantic import BaseModel


class Paper(BaseModel):
    arxiv_id: str
    title: str
    abstract: str
    authors: str
    url: str
    published: str
    updated: str
    categories: list[str]
    retrieved_date: str
    subtopic: str | None

    def __init__(self, **kwargs):
        kwargs["categories"] = kwargs["categories"].split(", ")

        optional_keys = ["subtopic"]
        for key in optional_keys:
            if key not in kwargs or (kwargs[key] == "" or kwargs[key] == "nan"):
                kwargs[key] = None

        super().__init__(**kwargs)
