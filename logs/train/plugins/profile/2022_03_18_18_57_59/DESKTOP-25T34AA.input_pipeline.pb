	??QI??V@??QI??V@!??QI??V@	??$??e????$??e??!??$??e??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??QI??V@Ǻ?????A??x?&?V@Y?q??????*	     @H@2F
Iterator::ModeljM????!?,O"ӰS@)?A`??"??1?/?~?QK@:Preprocessing2P
Iterator::Model::Prefetchg??j+???!tT???8@)g??j+???1tT???8@:Preprocessing2]
&Iterator::Model::Prefetch::MemoryCache?0?*??!?L?v?<5@)??~j?t??1(?i?n?3@:Preprocessing2a
*Iterator::Model::Prefetch::MemoryCacheImpl-C??6J?!???Id??)-C??6J?1???Id??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??$??e??#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Ǻ?????Ǻ?????!Ǻ?????      ??!       "      ??!       *      ??!       2	??x?&?V@??x?&?V@!??x?&?V@:      ??!       B      ??!       J	?q???????q??????!?q??????R      ??!       Z	?q???????q??????!?q??????JCPU_ONLYY??$??e??b 