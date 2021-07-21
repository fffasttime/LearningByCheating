/*
Return error insert result of data.
This code should be compiled to .so library for Python call.
*/

#include <assert.h>

float insert_float(float data, int bit){
    assert(bit<32);
    int *p=(void*)&data;
    (*p)^=1<<bit;
    return data;
}

double insert_double(double data, int bit){
    assert(bit<64);
    unsigned long long *p=(void*)&data;
    (*p)^=1ull<<bit;
    return data;
}

char insert_int8(char data, int bit){
    assert(bit<8);
    data^=1<<bit;
    return data;
}

short insert_int16(short data, int bit){
    assert(bit<16);
    data^=1<<bit;
    return data;
}

int insert_int32(int data, int bit){
    assert(bit<32);
    data^=1<<bit;
    return data;
}

long long insert_int64(long long data, int bit){
    assert(bit<64);
    data^=1ull<<bit;
    return data;
}

void insert_array(char *data, int bit){
    data[bit/8]^=1<<(bit%8);
}

#include <stdio.h>
int main(){
    float x;
    scanf("%f",&x);
    while (1){
        int bit; scanf("%d",&bit);
        x=insert_float(x, bit);
        printf("%e\n", x);
    }
}
