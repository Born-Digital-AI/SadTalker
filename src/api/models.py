from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field


class PreprocessType(str, Enum):
    resize = "resize"
    full = "full"
    crop = "crop"
    extcrop = "extcrop"
    extfull = "extfull"


class GenerateRequest(BaseModel):
    source_image: str = Field(..., description="base64 string of the source image")
    bg_image: Optional[str] = Field(None, description="base64 string of the background image")
    email: str = Field(..., description="email where the link to the generated avatar will be sent")
    preprocess_type: PreprocessType
    is_still_mode: bool
    exp_scale: float
