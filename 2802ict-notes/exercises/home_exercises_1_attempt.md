## AI and Agents

1. Based on thinking vs acting and humanly vs rationally, what are the four major views in Artificial Intelligence? Name and explain each of them. Which approach is currently taken by most AI researchers?


Thinking is about internal reasoning or cognition.
Acting is about external behaviour or choice of action.

Humanly means matching humans (how they think or behave).
Rationally means optimally given goals and information (doing “the right thing”), not necessarily human-like.


We have:

**~~Thinking humanly~~** (~~cognitive modelling~~)
Where a system thinks like a human including neurological processes.

**~~Thinking rationally~~** (~~Laws of thought~~)
Where a system reasons according to ideal principles of logic, ie propositional logic.

**~~Acting humanly~~** (~~turing test~~)
Where a system acts like a human.

**~~Acting rationally~~**
Where a system takes the best action to achieve their goals given their knowledge and uncertainty.
- The approach taken by researchers.

2. What is the Turing test? What are its contributions and problems? Which of the four major approaches in AI does it take? What is your comment on Turing test? 

Turing defined intelligent behavior as the ability to achieve human-level performance in all cognitive tasks, sufficient to fool an interrogator. It mirrors the acting humanly approach.

3. Name two most significant achievements in AI and explain why they are significant.

Alexnet showed that machine learning could outperform other computervision approaches.

ChatGPT showed that the transformer architecture could produce machines that act humanly.

4. What is an agent? What do you mean by a rational agent? 

An agent is a machine that perceives its environment through percepts and takes actions.

A rational agent is an agent that for each possible percept history chooses the action expected to maximise its performance measure.

## Search Strategies

1. What are the four desirable properties of search algorithms?  Describe what is 
meant by each of these four properties.

- Complete. Finds a solution where one exists.
- Optimal. Finds optimal solution where one exists.
- Space complexity. Asymptotic space used to complete search.
- Time complexity. Asymptotic time to complete search.

3. Breadth-first,  depth-first  and  uniform-cost  search  are  all  considered  to  be 
uninformed search strategies. 
1) Explain  using  words  and  diagrams  how  each  of  these  three  search 
mechanisms operates. 
2) Why are these algorithms considered uninformed? 
3) Describe iterative-deepening depth-first search, in terms of its relationship 
with  depth-first  search.  What  are  the  principal  advantages  of  iterative-
deepening depth-first search over depth-first search. 

1) BFS searches nodes using a FIFO queue, therefore searching nodes from nearest to furthest. DFS searches nodes using a LIFO queue, therefore searching nodes in a depth-first traversal pattern. The last node to be added to the queue will be expanded next. Uniform-cost-search using a priority queue ordered by the current path cost to the node.
2) They are considered uninformed as they do not possess a heuristic function to guide search.
3) Iterative-deepening depth-first search is type of search which repeatedly calls DFS where each call has a maximum search depth. The principal advantages of iterative-deepening depth-first search are (1) low memory usage (2) completeness (where solution fits in memory) and (3) optimality.


4. Does each of the following statements holds or not? If yes, explain why; if not, 
give a counterexample: 
1) Breadth-first search is a special case of uniform-cost search. 
2) Depth-first search is a special case of best-first search. 
3) Uniform-cost search is a special case of A* search.

1) UCS (Dijkstra) chooses the action that minimises the current path cost. BFS is a special case of UCS where each edge has a cost of 1.
2) ~~Best-first-search is a generic term for a search algorithm that uses a priority queue. DFS is not a special case of BFS.~~ DFS can be a special case of best-first-search if the evaluation function is the negative node depth.
3) A* is a best-first-search that utilises a cost function f where f = g (path cost) + h (hueristic cost). UCS can be thought of a special case of A* search where the heuristic function returns 0.

5. Todo

6. A* search is the most well-known informed search strategy.  
1) Outline the A* search algorithm.                                                      
2) In  general  A*  search  is  not  optimal.  Give  a  sufficient  condition  for  the 
optimality of A* search.        
3) Trace  the  operation  of  A*  search  applied  to  the  problem  of  getting  to 
Bucharest  from  Lugoj  using  the  straight-line  distance  heuristic.  That  is, 
show the sequence of nodes that the algorithm will consider and the f, g, 
and h score for each node.           
4) A robot wishes to move from square 0 (“START”) to square 22 (“GOAL”) 
in Figure 2. The following rules apply:

1) A* is a best-first-search where the evaluation function f incorporates current path cost (g) and a heuristic to guide search (h) (f = g + h).
2) A* is optimal when the heuristic is admissible. (Consistency is an event stronger guarantee where node goal exploration is guaranteed to be optimal and it is not necessary to keep track of cost-minising solutions until the current cost-minimal solution is less than or equal to all nodes currently in the frontier. Expanded nodes do not need to be reopened)