

void loop_for()
{
    // Loop with block
    for (int i = 0; i < 10; i++)
    {
        ;
    }

    // Loop without block
    for (int i = 0; i < 10; i++)
        ;
}

void loop_while()
{
    // Loop with block
    while (1)
    {
        ;
    }

    // Loop without block
    while (1)
        ;
}

void conditional_if()
{
    // If with block
    if (1)
    {
        ;
    }
    else
    {
        ;
    }

    // If without block
    if (1)
        ;
    else
        ;

    // If with else if
    if (1)
    {
        ;
    }
    else if (2)
    {
        ;
    }
    else
    {
        ;
    }
}

void conditional_switch()
{
    // Switch with block
    switch (1)
    {
    case 1:;
        break;
    case 2:;
        break;
    default:;
    }

    // Switch without block
    switch (1)
    {
    case 1:;
        break;
    case 2:;
        break;
    default:;
    }
}

void loop_conditional()
{
    // Loop with conditional
    for (int i = 0; i < 10; i++)
    {
        if (i % 2 == 0)
        {
            continue;
        }
        else
        {
            break;
        }
    }
}

void unreachable_code()
{
    // Unreachable code after return
    return;
    ;

    // Unreachable code after break
    for (int i = 0; i < 10; i++)
    {
        break;
        ;
    }
}

void error()
{
    /* Demonstration of tree sitter error recovery */
    function()
        function2();
}

int main()
{
    loop_for();
    loop_while();
    conditional_if();
    conditional_switch();
    loop_conditional();
    unreachable_code();
    error();
    return 0;
}