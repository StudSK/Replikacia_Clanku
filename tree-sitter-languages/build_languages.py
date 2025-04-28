from tree_sitter import Language

Language.build_library(
    'build/my-languages.so',  # Куда сохранить скомпилированную библиотеку
    [
        'tree-sitter-python',  # Где исходники грамматики
    ]
)