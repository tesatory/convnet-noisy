#include <noiselayer.cuh>
#include <assert.h>

void prob_project_dual(NVMatrix& m, NVMatrix& m_test) {
	int dim = m.getNumRows();
	assert(2 * dim == m.getNumCols());

	// normalize only the second half
	NVMatrix* m_noisy = new NVMatrix(dim, dim);
	m.sliceCols(dim, 2 * dim, *m_noisy);
	prob_project(*m_noisy);
	m_noisy->copy(m, 0, dim, 0, dim, 0, dim);
	delete m_noisy;

	// make the first half identity
	NVMatrix* m_clear =  new NVMatrix(dim, dim);
	m_test.sliceCols(0, dim, *m_clear); 
	m_clear->copy(m, 0, dim, 0, dim, 0, 0);
	delete m_clear;
}

// normalize so that each column represents probability using projection
void prob_project(NVMatrix& m) {
	const float eps = 0.000001;
	NVMatrix* pos = new NVMatrix(m);	// identify postive elements
	NVMatrix* x = new NVMatrix(m);
	pos->apply(NVMatrixOps::One());
	NVMatrix* r = new NVMatrix(m.getNumRows(), 1);
	NVMatrix* q = new NVMatrix(*r);
	while (true) {
		m.sum(1, *r);
		r->addScalar(-1);
		r->apply(NVMatrixOps::Abs(), *q);
		q->biggerThanScalar(eps);
		if (q->sum() ==  0) break;
		r->eltwiseDivide(pos->sum(1));
		m.addVector(*r, -1, *x);
		x->eltwiseMult(*pos);
		// pos->eltwiseMultByVector(*q);
		// x->eltwiseMultByVector(*q);
		pos->smallerThanScalar(0.5);
		m.eltwiseMult(*pos);
		m.add(*x);
		m.biggerThanScalar(0, *pos);
		// pos->smallerThanScalar(0.5);
		// pos->eltwiseMultByVector(*q);
		// pos->smallerThanScalar(0.5);
		m.eltwiseMult(*pos);
	}
	delete pos;
	delete x;
	delete r;
	delete q;
}		