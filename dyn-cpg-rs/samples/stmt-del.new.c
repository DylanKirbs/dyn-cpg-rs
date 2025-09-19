int main(int argc, char *argv[])
{
    int x = source();
    if (x > 0)
    {
        int z = 2 * x;
        sink(z);
    }
    return 0;
}