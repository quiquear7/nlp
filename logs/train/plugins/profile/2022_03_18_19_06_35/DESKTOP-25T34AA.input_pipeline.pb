	?):??C]@?):??C]@!?):??C]@	eD?????eD?????!eD?????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?):??C]@?A?f????A/?$9]@Yvq?-??*	     ?P@2]
&Iterator::Model::Prefetch::MemoryCache??j+????!g???1?E@)?!??u???1Ez?rvE@:Preprocessing2F
Iterator::Model????ׁ??!?HT?nL@)???S㥛?1??~5&D@:Preprocessing2P
Iterator::Model::PrefetchA??ǘ???!g???1?0@)A??ǘ???1g???1?0@:Preprocessing2a
*Iterator::Model::Prefetch::MemoryCacheImplǺ???F?!M?*g???)Ǻ???F?1M?*g???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9eD?????#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?A?f?????A?f????!?A?f????      ??!       "      ??!       *      ??!       2	/?$9]@/?$9]@!/?$9]@:      ??!       B      ??!       J	vq?-??vq?-??!vq?-??R      ??!       Z	vq?-??vq?-??!vq?-??JCPU_ONLYYeD?????b 