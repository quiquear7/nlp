	|a2U?W@|a2U?W@!|a2U?W@	??0+|?????0+|???!??0+|???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$|a2U?W@!?lV}??A??K7??W@Ya??+e??*	     @C@2F
Iterator::Model?ZӼ???!(?Y?	qR@)M?O???1`???;J@:Preprocessing2]
&Iterator::Model::Prefetch::MemoryCacheM?O???!`???;:@)Έ?????1??g?'8@:Preprocessing2P
Iterator::Model::Prefetch	?^)ˀ?!?15?wL5@)	?^)ˀ?1?15?wL5@:Preprocessing2a
*Iterator::Model::Prefetch::MemoryCacheImpl-C??6J?!V~B??? @)-C??6J?1V~B??? @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??0+|???#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	!?lV}??!?lV}??!!?lV}??      ??!       "      ??!       *      ??!       2	??K7??W@??K7??W@!??K7??W@:      ??!       B      ??!       J	a??+e??a??+e??!a??+e??R      ??!       Z	a??+e??a??+e??!a??+e??JCPU_ONLYY??0+|???b 