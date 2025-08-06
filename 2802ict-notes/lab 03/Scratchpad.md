1. ()
MRV assessment: all nodes have the same domain. (no preference)
DH assessment: SA has the most constrains, pick SA.
LCV assessment: Does not matter, choose red.
Forward chaining: WA, NT, Q, NSW and V cannot be red.

2. ((SA, red))
MRV assessment: choose one of WA, NT, Q, NSW and V.
DH: pick NT (of NT or Q or NSW)
LCV: Does not matter, choose blue
Forward chaining: WA, Q cannot be blue

3. ((SA, red), (NT, blue))
MRV assessment: choose one of WA or Q
DH: Q
LCV: Does not matter, choose green
Forward chaining: NSW cannot be green

4. ((SA, red), (NT, blue), (Q, green))
MRV assessment: choose one of WA or NSW
DH: pick NSW
LCV: Does not matter, choose blue
Forward chaining: V cannot be blue

5. ((SA, red), (NT, blue), (Q, green), (NSW, blue))
MRV assessment: choose one of WA or V
DH: pick WA (of WA or V)
LCV: Does not matter, choose green
Forward chaining: Does not matter

6. ((SA, red), (NT, blue), (Q, green), (NSW, blue), (WA, green))
MRV assessment: choose V
DH: Does not matter
LCV: Does not matter, choose green
Forward chaining: Does not matter

7. ((SA, red), (NT, blue), (Q, green), (NSW, blue), (WA, green), (V, green))
MRV assessment: choose T
DH: Does not matter
LCV: Does not matter, choose blue
Forward chaining: Does not matter

8. ((SA, red), (NT, blue), (Q, green), (NSW, blue), (WA, green), (V, green), (T, blue))

Solution found!