Find the definition location of a symbol at a specific position in a file.

This tool uses the language server to locate where a symbol (variable, function, class, etc.) is defined. It's particularly useful for navigating codebases and understanding the structure of the code.

The tool requires a file path and the exact line and column position of the symbol you want to look up.

<example>
To find the definition of a symbol at line 42, column 15 in a file:
{
    "path": "src/main.rs",
    "line": 42,
    "column": 15
}
</example>

<example>
To find where a function is defined when you see it being called:
{
    "path": "backend/api/handlers.py",
    "line": 128,
    "column": 20
}
</example>

<guidelines>
- Line and column numbers are 1-based (starting from 1, not 0)
- The position should be on or within the symbol name for best results
- This tool works best with symbols that have clear definitions (functions, classes, variables)
- Some symbols may have multiple definitions (e.g., overloaded functions)
- If no definition is found, the symbol might be from an external library or not properly indexed
</guidelines>
