int main(int argc, char *argv[])
{
    int x = source();
    if (x > 0)
    {
        int y = 2 * x;
        sink(y);
    }
    return 0;
}