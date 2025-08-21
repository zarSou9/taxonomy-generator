from pydantic import BaseModel


class ArxivCategoryInfo(BaseModel):
    category_group: str
    code: str
    name: str
    description: str
