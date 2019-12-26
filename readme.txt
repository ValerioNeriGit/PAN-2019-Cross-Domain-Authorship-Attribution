esempio di parametri da command line:
    -i /opt/projects/attribution/src/Datasets/training-dataset-2019-01-23 -o /opt/projects/attribution/src/Code/baseline/my_outs

moduli necessari:
    cython
    sklearn
    scipy
    numpy
    tabulate
    nltk

Nota:
    1.i pickle sono in baseline/pickles potrebbe essere necessario rimuoverli e ri-generarli.
    2. Se runnato con pycharm inserire  come source root sia Code che tools

il progrmamma è strutturato come segue:
    classifier.py xor classifier_semplified.py sono i file principali
    std.py è il file in cui sono contenute tutte le funzioni utili
    std_c.pyx è un file cython in cui sono contenute poche funzioni alcune features che non sono utilizzate
        si compila automaticamente quando lanciato classifier.py xor classifier_semplified.py

    ref e resources contengono alcuni file utili per alcune prove, principalmente function_word.py contiene diverse variabili (oltre alle fw) utili per certi preprocessing

classifier_semplified.py contiene dei commenti che ne indicano il funzionamento, in generale:
    esegue il parsing delle command line option e esegue main(..)

    main:
        esegue del preprocessing semplice necessario per avere ogni problema ben configurato,
        crea la pool per l'esecuzione parallela e lancia "evaluate_problem" ogni problema in parallelo
        Nota: dato che i file soluzione sono salvati da evaluate_problem non è necessario che ci sia un valore di ritorno

    evaluate_problem:
        raccoglie i dati del problema e i documento di train e test (o meglio dev)
        chiama "vectorization" la quale genererà le features e ritorna una lista di matrici
        definisce clfs come una lista di classificatori, in modo che fit and predict the i-th classifier in clfs with the i-th matrix (features)
            of train_data and test_data returned by "vectorization"
        esegue lo scaling
        chiama "multi_classification" che eseguirà fit, predict_proba su (for clf, train_data, test_data in zip(clfs, train_datas, test_datas))
        utilizza reject_option per valutare gli unknown
        salva i file di soluzione

    vectorization:
        creata le varie features
        ritorna due liste di matrici una per il train ed una per il test

    multi_claffification:
        prende in input una lista di classificatori, una lista di matrici di features ed una lista di matrici di test,
        per ogni i-esimo classificatore esegue fit su i-esima matrice di train e predict_proba su i-esima matrice di testa
            dato che viene usata la funzione zip di python per fare questo in caso le dimensioni delle liste clf, trains, tests siano diverse, verranno eseguite
            iterazioni pari alla dimensione del più piccolo
        esegue softvoting fra le predizioni e le probabilità dei classificatori
            È possibile utilizzare la rete neurale come ensamble method utilizzando la funzione "porbabilities_clf" invece del softvoting e decommentando alcune
            istruzzioni.

         ritona una lista con un solo elemento (per motivi storici) di predizioni ed una lista con un solo elemento (per motivi storici) di probabilità
            (necessarie per poter fare una threshold per gli unknown)

