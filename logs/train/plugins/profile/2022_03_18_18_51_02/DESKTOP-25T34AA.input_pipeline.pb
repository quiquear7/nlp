	T㥛?dW@T㥛?dW@!T㥛?dW@	CES?)\??CES?)\??!CES?)\??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$T㥛?dW@??ܵ??A?????VW@Yu????*?????G@)       =2F
Iterator::Model	??g????!?2^??U@)2??%䃞?1???v# P@:Preprocessing2P
Iterator::Model::Prefetch?&S???!??&?l?3@)?&S???1??&?l?3@:Preprocessing2]
&Iterator::Model::Prefetch::MemoryCache?<,Ԛ?}?!-h?/@)S?!?uq{?1o??-@:Preprocessing2a
*Iterator::Model::Prefetch::MemoryCacheImpla2U0*?C?!?S{???)a2U0*?C?1?S{???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9BES?)\??#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??ܵ????ܵ??!??ܵ??      ??!       "      ??!       *      ??!       2	?????VW@?????VW@!?????VW@:      ??!       B      ??!       J	u????u????!u????R      ??!       Z	u????u????!u????JCPU_ONLYYBES?)\??b 