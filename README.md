# Perceptron-and-MLP-classifier

normalize()
-
returneaza datele de antrenare si pe cele de testare normalizate prin scaderea mediei si impartirea la deviatia standard, folosind modelul StandardScaler din libraria scikit-learn.

test_train_perceptron()
-
- metoda antreaza un percepton pana cand eroare nu se imbunatateste cu 1e-5 fata de epocile anterioare, cu rata de invatare 0.1, folosind clasificator Perceptron din libraria scikit-learn;
- afiseaza acuratetea pe mutimea de antrenare si pe cea de testare folosind atributul score al clasificatorului Perceptron; 
- afiseaza ponderile, bias-ul si numarul de epoci parcurse pana la convergenta folosind atributele coef_ , intercept_ si respectiv n_iter_ ale clasificatorului;
- afiseaza grafic planul de decizie al clasificatorului.

test_train(classifier)
-
- metoda antreneaza o retea de perceptroni folosind clasificatorul classifier de tip MLPClassifier din libraria scikit-learn;
- afiseaza acuratetea pe multimea de antrenare si pe cea de testare folosind atributul score al clasificatorului; 
- afiseaza numarul de epoci parcurse pana la convergenta folosind atribututul n_iter_ al clasificatorului de tip MLPCLassifier.
 

