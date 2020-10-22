
License Note: 
	The source code of our model is build on top of the source code of
	Learn to Explain source as a starting point. 


Instruction runs:
		All these code should be written in cmd on the folder file for each 
		specific dataset. 
	Mnist:	One can change the number of groups and number of important groups in the code.
	num_groups, num_important_groups. adn the BATCH_SIZE. 
			
		-->python gI_Mnist.py --train --epochs 10

	F-Mnist: One can change the number of groups and number of important groups in the code.
	num_groups, num_important_groups. adn the BATCH_SIZE. 
		
		-->python Fashion_final_2.py --train --epochs 10
		
	Syn:
		-->python gI_syn.py --train --datatype XOR

			gI is the model that choose the important groups as well.

		-->python gC_syn.py --train --datatype XOR
		
			gC just finds the groups without the important groups. 


			The relationship with the paper:
			XOR = R1
			orange_skin = R2
			nonlinear_additive =R3

			For changing the data structure to D1 D2 D3 or D1+D2 as it mentioned in the paper,
			One has to change the make_data.py file to generate thoes results. 






