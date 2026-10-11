"""Tool package manifest (design doc 05, 2.1)."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# 05 doc 2.3 draft allowlist. Additions go through a governance `policy` proposal.
ALLOWED_PACKAGES = {"jsonschema", "numpy", "pandas", "sympy", "beautifulsoup4", "pyyaml",
                    "python-dateutil", "regex"}
NAME = re.compile(r"^[a-z][a-z0-9-]{2,39}$")


class Permissions(BaseModel):
    network: Literal["none"] = "none"  # v1: tools never get the network
    storage_mb: int = Field(default=0, ge=0, le=0)


class Limits(BaseModel):
    timeout_s: int = Field(default=10, ge=1, le=30)
    memory_mb: int = Field(default=256, ge=32, le=512)


class ToolManifest(BaseModel):
    name: str = Field(description="소문자, 숫자, '-' (3~40자). 도구 ID는 '<노드>.<name>'이 된다.")
    description: str = Field(min_length=10, max_length=500)
    input_schema: dict[str, Any] = Field(description="입력 JSON Schema (type: object)")
    output_schema: dict[str, Any] = Field(description="출력 JSON Schema (type: object)")
    permissions: Permissions = Field(default_factory=Permissions)
    limits: Limits = Field(default_factory=Limits)
    python_packages: list[str] = Field(default_factory=list, max_length=5)

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not NAME.match(v):
            raise ValueError("name은 소문자로 시작하고 소문자·숫자·'-'만, 3~40자")
        return v

    @field_validator("input_schema", "output_schema")
    @classmethod
    def _object_schema(cls, v: dict[str, Any]) -> dict[str, Any]:
        if v.get("type") != "object":
            raise ValueError("schema는 {\"type\": \"object\", ...} 형태여야 한다")
        return v

    @field_validator("python_packages")
    @classmethod
    def _packages(cls, v: list[str]) -> list[str]:
        bad = sorted(set(v) - ALLOWED_PACKAGES)
        if bad:
            raise ValueError(f"허용되지 않은 패키지 {bad}. 허용 목록: {sorted(ALLOWED_PACKAGES)}")
        return v


class ToolTest(BaseModel):
    input: dict[str, Any]
    expected: dict[str, Any] = Field(description="출력에 반드시 들어 있어야 하는 키와 값")
