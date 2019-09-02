# Contributing to PyWarm

[PyWarm](https://github.com/blue-season/pywarm) is developed on [GitHub](https://github.com/blue-season/pywarm). 

Please use GitHub to file Bug reports and submit pull requests. Please document and test before submissions.

PyWarm is developed with Python 3.7, but has been tested to work with Python 3.6+.

## Coding Style

You may have noticed that the source code of PyWarm uses some distinct style conventions.

### PEP 8

All guidelines here are in addition to or an upgrade of the Python [PEP 8](https://www.python.org/dev/peps/pep-0008/).

### Max Line Length

120 characters. 79 is too short. PyWarm is developed on a 49" 5120 x 1440 monitor.

### Closing Parentheses

Put closing parentheses at the same level of the last object. Do not put them on a new line.

```Python tab="Yes"
x = dict(a=1, b=2,
    c=3, d=4, ) # closing parentheses same line, yes
```

```Python tab="No"
x = dict(a=1, b=2,
    c=3, d=4
) # closing parentheses separate line, no
```

### Indentation

Indent 1 level (4 spaces) for line continuation, or 2 levels to distinguish from the next line.
Never align with opening delimiter:

```Python tab="Yes"
foo = long_function_name(
    var_one=1, var_two=2, var_three=3, var_four=4, ) # 1 level indent, yes
```

```Python tab="No"
# what if the function name plus indent is really long, like 70 characters?
foo = long_function_name(var_one=1, var_two=2,
                         var_three=3, var_four=4) # align, no
```

### Blank Lines

Completely avoid blank lines inside function and methods.
Instead, organize / refactor the code to be cleaner and shorter.
If there is a really strong need, use an empty comment at the same level of indentation.

### Type Annotations

Don't use. Instead, write detailed type information in the docstring.

### Inline comments

Keep them to a minimum.
Comments should reflect intentions, not echo implementaions.

### String Quotes

In code, use single-quote `''` whenever possible. For docstring, use triple double quotes: `""" """`.

### Naming Conventions

If you don't want others to mess around with an object, add a single underscore before its name:

```Python
_do_not_mess_with_me
```

If a function / method alters its inputs, add a single underscore after it:

```Python
i_mess_with_inputs_(x) # content of x will change after each call
```
